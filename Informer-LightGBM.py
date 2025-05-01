import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
from datetime import datetime


# Informer Model Definition
class PositionalEncoding(nn.Module):
    """Position Encoding"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a PE matrix of sufficient length
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class ProbAttention(nn.Module):
    """Informer's Probabilistic Attention Mechanism"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Ensure the sampling quantity is legal
        sample_k = min(sample_k, L_K)
        n_top = min(n_top, L_Q)
        
        # Prevent empty tensors
        if sample_k <= 0:
            sample_k = 1
        if n_top <= 0:
            n_top = 1

        # Calculate sparse sorting TopK Query sampling
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # Random sampling, size [L_Q, sample_k]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # [B, H, L_Q, sample_k, D]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # [B, H, L_Q, sample_k]

        # Find the most relevant key
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # [B, H, L_Q]
        M_top = M.topk(n_top, sorted=False)[1]  # [B, H, n_top]

        # Use mask for sparse attention
        Q_reduce = Q[torch.arange(B)[:, None, None], 
                     torch.arange(H)[None, :, None], 
                     M_top, :]  # [B, H, n_top, D]
        
        # Normal attention calculation
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # [B, H, n_top, L_K]

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1])
        else:  # Use 0
            contex = torch.zeros(B, H, L_Q, V.shape[-1]).to(V.device)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1)  # scores: [B, H, n_top, L_K]

        context_in = context_in.clone()

        context_in[torch.arange(B)[:, None, None], 
                   torch.arange(H)[None, :, None], 
                   index, :] = torch.matmul(attn, V)  # [B, H, n_top, D]
        
        if self.output_attention:
            attns = torch.ones(B, H, L_Q, L_V).to(V.device) * 0
            attns[torch.arange(B)[:, None, None], 
                 torch.arange(H)[None, :, None], 
                 index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(1, 2)  # [B, H, L_Q, D]
        keys = keys.transpose(1, 2)  # [B, H, L_K, D]
        values = values.transpose(1, 2)  # [B, H, L_K, D]

        # Ensure L_Q and L_K are greater than 0, handle log(0) cases
        L_K = max(1, L_K)
        L_Q = max(1, L_Q)
        
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # Number of sampled queries
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # Number of sampled queries

        U_part = min(U_part, L_K)
        u = min(u, L_Q)
        
        # Ensure U_part and u are at least 1
        U_part = max(1, U_part)
        u = max(1, u)

        scores_top, index = self._prob_QK(queries, keys, u, U_part)
        
        # Add scale
        scale = self.scale or 1.0 / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
            
        # Add mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores_top.masked_fill_(attn_mask, -float('inf'))
            else:
                scores_top += attn_mask

        # Get context
        context = self._get_initial_context(values, L_Q)  # [B, H, L_Q, D]
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(1, 2), attn


class AttentionLayer(nn.Module):
    """Attention Layer Wrapper"""
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project and reshape
        queries = self.query_projection(queries).reshape(B, L, H, -1)
        keys = self.key_projection(keys).reshape(B, S, H, -1)
        values = self.value_projection(values).reshape(B, S, H, -1)

        # Calculate attention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """Informer Encoder Layer"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None):
        # Self-attention
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Feed-forward network
        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(1, 2)
        
        return self.norm2(x + y), attn


class ConvLayer(nn.Module):
    """Convolutional layers used for downsampling"""
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class Encoder(nn.Module):
    """Informer Encoder"""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """Informer Decoder Layer"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention
        x_attn, self_attn = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(x_attn)
        x = self.norm1(x)
        
        # Cross-attention
        x_cross, cross_attn = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = x + self.dropout(x_cross)
        x = self.norm2(x)
        
        # Feed-forward network
        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(1, 2)
        
        return self.norm3(x + y), self_attn, cross_attn


class Decoder(nn.Module):
    """Informer Decoder"""
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class InformerModel(nn.Module):
    """Complete Informer model"""
    def __init__(self, input_dim, calendar_dim, enc_in, dec_in, 
                 d_model=512, n_heads=8, d_ff=2048,
                 e_layers=3, d_layers=2, 
                 dropout=0.1, activation='gelu',
                 output_attention=False, factor=5):
        super(InformerModel, self).__init__()
        self.output_attention = output_attention
        # Encoder and decoder input projection
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.calendar_embedding = nn.Linear(calendar_dim, d_model)
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Informer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers-1)
            ] if e_layers > 1 else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Informer解码器
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # 最终的预测层
        self.projection = nn.Linear(d_model, 1, bias=True)
        
    def forward(self, x, calendar_features, x_dec=None, calendar_dec=None):
        # x: [Batch, seq_len, input_dim]
        # calendar_features: [Batch, seq_len, calendar_dim]
        
        # Embed and position encode encoder input
        x_embed = self.input_embedding(x)
        cal_embed = self.calendar_embedding(calendar_features)
        enc_input = x_embed + cal_embed
        enc_input = self.pos_encoder(enc_input)
        
        # Encoder forward propagation
        enc_output, attns = self.encoder(enc_input)
        
        # If there is no decoder input, use the last time step of the encoder as the decoder input
        if x_dec is None:
            x_dec = x[:, -1:, :]
            calendar_dec = calendar_features[:, -1:, :]
        

        x_dec_embed = self.input_embedding(x_dec)
        cal_dec_embed = self.calendar_embedding(calendar_dec)
        dec_input = x_dec_embed + cal_dec_embed
        dec_input = self.pos_encoder(dec_input)
        
        # Decoder forward propagation
        dec_output = self.decoder(dec_input, enc_output)
        
        # Final prediction layer
        output = self.projection(dec_output)
        
        if self.output_attention:
            return output, attns
        else:
            return output


# Informer Model Training and Prediction
class InformerPredictor:
    def __init__(self, feature_dim, seq_len, pred_len, d_model=512, n_heads=8, e_layers=2, d_layers=1, dropout=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.dropout = dropout
        
        # Informer Model Initialization
        self.model = Informer(
            enc_in=self.feature_dim,
            dec_in=self.feature_dim,
            c_out=1,
            seq_len=self.seq_len,
            label_len=self.seq_len//2,
            out_len=self.pred_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            dropout=self.dropout
        ).to(self.device)
        self.criterion = nn.MSELoss()  
        
    def bayesian_optimize(self, x_train, y_train, x_val, y_val):
        """
        Use Bayesian optimization to optimize Informer model parameters
        """
        def objective(d_model, n_heads, e_layers, d_layers, dropout):
            # Build the model
            self.d_model = int(d_model)
            self.n_heads = int(n_heads)
            self.e_layers = int(e_layers)
            self.d_layers = int(d_layers)
            self.dropout = dropout
            
            # Reinitialize the Informer model
            self.model = Informer(
                enc_in=self.feature_dim,
                dec_in=self.feature_dim,
                c_out=1,
                seq_len=self.seq_len,
                label_len=self.seq_len//2,
                out_len=self.pred_len,
                d_model=self.d_model,
                n_heads=self.n_heads,
                e_layers=self.e_layers,
                d_layers=self.d_layers,
                dropout=self.dropout
            ).to(self.device)
            
            # Validate
            val_loss = self._validate_model(x_val, y_val)
            return -val_loss  # Bayesian optimization seeks maximum value, loss is negative
        
        # Define parameter range
        pbounds = {
            'd_model': (32, 512),
            'n_heads': (4, 16),
            'e_layers': (2, 4),
            'd_layers': (1, 3),
            'dropout': (0.0, 0.3)
        }
        
        # Execute Bayesian optimization
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42
        )
        
        optimizer.maximize(init_points=5, n_iter=20)
        
        # Return the best parameters
        best_params = optimizer.max['params']
        # Convert parameters to integers
        best_params['d_model'] = int(best_params['d_model'])
        best_params['n_heads'] = int(best_params['n_heads'])
        best_params['e_layers'] = int(best_params['e_layers'])
        best_params['d_layers'] = int(best_params['d_layers'])
        
        print(f"Informer Best Parameters: {best_params}")
        return best_params
        
    def _train_model(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=32, early_stopping=10):
        """
        Train the Informer model
        """
        # Create data loader
        train_dataset = InformerDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation set data loader
        val_dataset = InformerDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward propagation
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Backward propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                
                val_loss /= len(val_loader)
            
                # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                # torch.save(self.model.state_dict(), 'best_informer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping at the {epoch+1}/{epochs} epoch")
                    break
            
            print(f"Epoch {epoch+1}/{epochs}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
        
        # Load the best model
        # self.model.load_state_dict(torch.load('best_informer_model.pth'))
        
        return best_val_loss
    
    def _validate_model(self, x_val, y_val):
        """
        Validate model performance
        """
        self.model.eval()
        val_dataset = InformerDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, x_train, y_train, x_val=None, y_val=None, batch_size=32, epochs=50, opt_params=None):
        """
        Train the Informer model
        """
        # If no validation set is provided, use part of the training set as validation
        if x_val is None or y_val is None:
            val_size = int(len(x_train) * 0.2)
            x_val, y_val = x_train[-val_size:], y_train[-val_size:]
            x_train, y_train = x_train[:-val_size], y_train[:-val_size]
        
        # If optimization parameters are provided, use them to build the model
        if opt_params:
            self.d_model = opt_params['d_model']
            self.n_heads = opt_params['n_heads']
            self.e_layers = opt_params['e_layers']
            self.d_layers = opt_params['d_layers']
            self.dropout = opt_params['dropout']
        
        # Initialize the Informer model
        self.model = Informer(
            enc_in=self.feature_dim,
            dec_in=self.feature_dim,
            c_out=1,
            seq_len=self.seq_len,
            label_len=self.seq_len//2,
            out_len=self.pred_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Train the model
        print("Start training Informer model...")
        self._train_model(x_train, y_train, x_val, y_val, epochs=epochs, batch_size=batch_size)
        
        return self
    
    def predict(self, x_test, rolling=True):
        """
        Use Informer model for prediction, support rolling prediction
        """
        self.model.eval()
        predictions = []
        
        if rolling:
            # Rolling prediction, each prediction will be added to the input matrix
            for i in range(len(x_test)):
                # Current time step input
                current_x = torch.FloatTensor(x_test[i:i+1]).to(self.device)
                
                # Prediction
                with torch.no_grad():
                    pred = self.model(current_x)
                    pred = pred.cpu().numpy()
                
                predictions.append(pred[0])
                
                # If there is a true value, update the input sequence
                if i < len(x_test) - 1 and self.pred_len == 1:
                    # Update the input sequence of the next time step
                    x_test[i+1, :-1] = x_test[i+1, 1:]  
                    x_test[i+1, -1] = pred[0, 0]  
        else:
            # Non-rolling prediction, predict all at once
            with torch.no_grad():
                test_dataset = InformerDataset(x_test, np.zeros((len(x_test), self.pred_len, 1)))
                test_loader = DataLoader(test_dataset, batch_size=64)
                
                for batch_x, _ in test_loader:
                    batch_x = batch_x.to(self.device)
                    preds = self.model(batch_x)
                    preds = preds.cpu().numpy()
                    
                    for pred in preds:
                        predictions.append(pred)
        
        return np.array(predictions)

# LightGBM Model Training and Prediction
class LightGBMPredictor:
    def __init__(self, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.models = []  # 存储多个预测时间步的模型
        
    def bayesian_optimize(self, x_train, y_train, time_indices=None):
        """
        Use Bayesian optimization algorithm to optimize LightGBM model parameters
        """
        best_params_list = []
        
        # Train a model for each prediction time step
        for i in range(self.pred_len):
            # Extract the target of the current time step
            y_train_step = y_train[:, i, 0]
            
            # Define the validation function
            def lgbm_evaluate(learning_rate, num_leaves, max_depth, feature_fraction, bagging_fraction):
                params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'learning_rate': learning_rate,
                    'num_leaves': int(num_leaves),
                    'max_depth': int(max_depth),
                    'feature_fraction': feature_fraction,
                    'bagging_fraction': bagging_fraction,
                    'verbose': -1
                }
                
                # Use time series cross-validation
                if time_indices is not None:
                    cv_scores = []
                    tscv = TimeSeriesSplit(n_splits=5)
                    for train_idx, val_idx in tscv.split(time_indices[:len(x_train)]):
                        # Split the training set and validation set
                        X_train_cv, X_val_cv = x_train[train_idx], x_train[val_idx]
                        y_train_cv, y_val_cv = y_train_step[train_idx], y_train_step[val_idx]
                        
                        # Train the model
                        lgb_train = lgbm.Dataset(X_train_cv, y_train_cv)
                        lgb_val = lgbm.Dataset(X_val_cv, y_val_cv, reference=lgb_train)
                        
                        model = lgbm.train(
                            params, 
                            lgb_train,
                            valid_sets=[lgb_val],
                            early_stopping_rounds=50,
                            num_boost_round=1000,
                            verbose_eval=False
                        )
                        
                        # Validate
                        y_pred = model.predict(X_val_cv)
                        mse = mean_squared_error(y_val_cv, y_pred)
                        cv_scores.append(mse)
                    
                    return -np.mean(cv_scores)  # 返回负MSE，贝叶斯优化寻找最大值
                else:
                    # Simple split training set and validation set
                    train_size = int(len(x_train) * 0.8)
                    X_train_cv, X_val_cv = x_train[:train_size], x_train[train_size:]
                    y_train_cv, y_val_cv = y_train_step[:train_size], y_train_step[train_size:]
                    
                    lgb_train = lgbm.Dataset(X_train_cv, y_train_cv)
                    lgb_val = lgbm.Dataset(X_val_cv, y_val_cv, reference=lgb_train)
                    
                    model = lgbm.train(
                        params, 
                        lgb_train,
                        valid_sets=[lgb_val],
                        early_stopping_rounds=50,
                        num_boost_round=1000,
                        verbose_eval=False
                    )
                    
                    y_pred = model.predict(X_val_cv)
                    mse = mean_squared_error(y_val_cv, y_pred)
                    return -mse  # 返回负MSE
            
            # Define parameter range
            pbounds = {
                'learning_rate': (0.001, 0.1),
                'num_leaves': (5, 100),
                'max_depth': (3, 15),
                'feature_fraction': (0.6, 1.0),
                'bagging_fraction': (0.6, 1.0)
            }
            
            # Execute Bayesian optimization
            optimizer = BayesianOptimization(
                f=lgbm_evaluate,
                pbounds=pbounds,
                random_state=42
            )
            
            optimizer.maximize(init_points=5, n_iter=20)
            
            # Store the best parameters
            best_params = optimizer.max['params']
            best_params['num_leaves'] = int(best_params['num_leaves'])
            best_params['max_depth'] = int(best_params['max_depth'])
            best_params_list.append(best_params)
            
            print(f"LightGBM model {i+1} best parameters: {best_params}")
        
        return best_params_list
    
    def train(self, x_train, y_train, time_indices=None, opt_params_list=None):
        """
        Train LightGBM model
        """
        # Clear existing models
        self.models = []
        
        # Train a model for each prediction time step
        for i in range(self.pred_len):
            print(f"Training LightGBM model {i+1}/{self.pred_len}...")
            
            # Extract the target of the current time step
            y_train_step = y_train[:, i, 0]
            
            # Use optimized parameters
            if opt_params_list and i < len(opt_params_list):
                params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'learning_rate': opt_params_list[i]['learning_rate'],
                    'num_leaves': int(opt_params_list[i]['num_leaves']),
                    'max_depth': int(opt_params_list[i]['max_depth']),
                    'feature_fraction': opt_params_list[i]['feature_fraction'],
                    'bagging_fraction': opt_params_list[i]['bagging_fraction'],
                    'verbose': -1
                }
            else:
                # Default parameters
                params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'learning_rate': 0.1,
                    'num_leaves': 31,
                    'max_depth': 7,
                    'verbose': -1
                }
            
            # Create dataset
            lgb_train = lgbm.Dataset(x_train, y_train_step)
            
            # Train the model
            model = lgbm.train(
                params,
                lgb_train,
                num_boost_round=1000,
                verbose_eval=100  # Print every 100 rounds
            )
            
            # Store the model
            self.models.append(model)
        
        return self
    
    def predict(self, x_test, rolling=True):
        """
        Use LightGBM model for prediction, support rolling prediction
        """
        predictions = []
        
        if rolling and self.pred_len == 1:
            # Rolling prediction, each prediction will be added to the input matrix
            for i in range(len(x_test)):
                # Current time step input
                current_x = x_test[i:i+1]
                
                # Predict for each time step
                step_preds = []
                for j, model in enumerate(self.models):
                    pred = model.predict(current_x)[0]
                    step_preds.append(pred)
                
                predictions.append(np.array(step_preds).reshape(-1, 1))
                
                # If there is a true value, update the input sequence
                if i < len(x_test) - 1:
                    # Update the input sequence of the next time step, include the true observation value
                    feature_dim = x_test.shape[2] // self.seq_len
                    next_x = x_test[i+1].reshape(self.seq_len, feature_dim)
                    next_x[:-1] = next_x[1:]  # Move data
                    next_x[-1] = np.array([step_preds[0]])  # Add the prediction value to the end
                    x_test[i+1] = next_x.reshape(-1)
        else:
            # Non-rolling prediction, predict all at once
            predictions = np.zeros((len(x_test), self.pred_len, 1))
            for i, model in enumerate(self.models):
                if i < self.pred_len:
                    preds = model.predict(x_test)
                    predictions[:, i, 0] = preds
        
        return np.array(predictions)

# Stacking集成模型
class StackingEnsemble:
    def __init__(self, pred_len):
        self.pred_len = pred_len
        # Use multiple linear regression as the meta model
        self.meta_models = [LinearRegression() for _ in range(pred_len)]
    
    def fit(self, predictions_list, y_true, time_indices=None):
        """
        Train Stacking ensemble model, use time series cross-validation
        Multiple linear regression parameter estimation uses the least squares method
        """
        if time_indices is None:
            time_indices = np.arange(len(y_true))
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Prepare training data for the meta model
        meta_features = np.zeros((len(y_true), len(predictions_list), self.pred_len))
        
        print("Generating meta features for Stacking...")
        # For each base model prediction result, use time series cross-validation to generate meta features
        for model_idx, model_name in enumerate(predictions_list.keys()):
            print(f"Processing {model_name} model...")
            for train_idx, val_idx in tscv.split(time_indices):
                # Train the base model
                if model_name == 'informer':
                    # Get the model and data
                    informer_model = predictions_list[model_name]['model']
                    x_train_fold = predictions_list[model_name]['x'][0][train_idx]
                    
                    # Train the base model
                    y_train_fold = y_true[train_idx]
                    informer_model.train(x_train_fold, y_train_fold, epochs=5)
                    
                    # Predict on the validation set
                    x_val_fold = predictions_list[model_name]['x'][0][val_idx]
                    val_preds = informer_model.predict(x_val_fold, rolling=False)
                    
                elif model_name == 'lightgbm':
                    # Get the model and data
                    lgbm_model = predictions_list[model_name]['model']
                    x_train_fold = predictions_list[model_name]['x'][0][train_idx]
                    
                    # Train the base model
                    y_train_fold = y_true[train_idx]
                    lgbm_model.train(x_train_fold, y_train_fold, epochs=5)
                    
                    # Predict on the validation set
                    x_val_fold = predictions_list[model_name]['x'][0][val_idx]
                    val_preds = lgbm_model.predict(x_val_fold, rolling=False)
                else:
                    continue
                
                # Store the prediction results as meta features
                for i, idx in enumerate(val_idx):
                    if idx < len(meta_features) and i < len(val_preds):
                        meta_features[idx, model_idx] = val_preds[i, :, 0]
        
        print("Training multiple linear regression meta model...")
        # Train the meta model - multiple linear regression
        for i in range(self.pred_len):
            # Extract specific time step meta features and target
            step_features = meta_features[:, :, i]
            step_targets = y_true[:, i, 0]
            
            # Train the multiple linear regression meta model
            # LinearRegression uses the least squares method to estimate the regression coefficients
            self.meta_models[i].fit(step_features, step_targets)
            
            # Print the regression coefficients
            coefficients = self.meta_models[i].coef_
            intercept = self.meta_models[i].intercept_
            print(f"Regression coefficients for step {i+1}: {coefficients}, intercept: {intercept}")
        
        return self
    
    def predict(self, base_predictions):
        """
        使用元模型对基模型的预测结果进行集成
        """
        # Extract all base model prediction results
        num_samples = base_predictions[list(base_predictions.keys())[0]].shape[0]
        meta_features = np.zeros((num_samples, len(base_predictions), self.pred_len))
        
        # Arrange base model prediction results as meta features
        for model_idx, model_name in enumerate(base_predictions.keys()):
            preds = base_predictions[model_name]
            for i in range(num_samples):
                meta_features[i, model_idx] = preds[i, :, 0]
        
        # Use the multiple linear regression meta model for final prediction
        ensemble_predictions = np.zeros((num_samples, self.pred_len, 1))
        for i in range(self.pred_len):
            step_features = meta_features[:, :, i]
            # Use the trained linear regression model to predict
            ensemble_predictions[:, i, 0] = self.meta_models[i].predict(step_features)
        
        return ensemble_predictions

# Evaluation function
def evaluate_predictions(y_true, y_pred, target_scaler=None):
    """
    Calculate evaluation metrics: MAPE, MAE, MSE and R²
    """
    if target_scaler:
        # Inverse normalization
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    
    # Flatten the data
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # Calculate various metrics
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Calculate MAPE, avoid division by zero error
    mask = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    
    return {
        'MAPE': mape,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }

# Plot prediction results
def plot_prediction(y_true, y_pred, title='负荷预测结果对比'):
    """
    Plot the comparison curve of true load and predicted load
    """
    plt.figure(figsize=(12, 6))
    
    # Flatten the data for plotting
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    plt.plot(y_true_flat, label='True load', markersize=4, linewidth=1)
    plt.plot(y_pred_flat, label='Predicted load', markersize=4, linewidth=1)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the image
    plt.savefig('load_forecast_result.png', dpi=300)
    plt.show()

# Main function
def main():
    # Parameter settings
    data_path = 'load_data.csv'
    seq_len = 24  
    pred_len = 1  

    train_start = 'Y-M-D'
    train_end = 'Y-M-D'
    test_start = 'Y-M-D'
    test_end = 'Y-M-D'
    
    # Load and preprocess data
    data = load_and_preprocess_data(
        data_path, 
        seq_len, 
        pred_len, 
        train_start=train_start, 
        train_end=train_end, 
        test_start=test_start, 
        test_end=test_end
    )
    
    # Initialize Informer model
    informer_model = InformerPredictor(
        feature_dim=data['x_train'].shape[2],
        seq_len=seq_len,
        pred_len=pred_len
    )
    
    # Initialize LightGBM model
    lgbm_model = LightGBMPredictor(seq_len, pred_len)
    
    # Use Bayesian optimization to optimize Informer model parameters
    print("Optimizing Informer model parameters...")
    informer_opt_params = informer_model.bayesian_optimize(
        data['x_train'], data['y_train'],
        data['x_train'][-100:], data['y_train'][-100:]
    )
    
    # Use Bayesian optimization to optimize LightGBM model parameters
    print("Optimizing LightGBM model parameters...")
    lgbm_opt_params = lgbm_model.bayesian_optimize(
        data['lgbm_x_train'], data['y_train'], data['train_time_indices'][:len(data['lgbm_x_train'])]
    )
    
    # Train Informer model
    informer_model.train(data['x_train'], data['y_train'], opt_params=informer_opt_params)
    
    # Train LightGBM model
    lgbm_model.train(data['lgbm_x_train'], data['y_train'], data['train_time_indices'][:len(data['lgbm_x_train'])], lgbm_opt_params)
    
    # Use Informer model for prediction
    informer_preds = informer_model.predict(data['x_test'], rolling=True)
    
    # Use LightGBM model for prediction
    lgbm_preds = lgbm_model.predict(data['lgbm_x_test'], rolling=True)
    
    # Ensemble prediction results
    stacking_model = StackingEnsemble(pred_len)
    
    # Prepare training data for the meta model
    base_predictions = {
        'informer': {
            'model': informer_model,
            'x': [data['x_train']]
        },
        'lightgbm': {
            'model': lgbm_model,
            'x': [data['lgbm_x_train']]
        }
    }
    
    # Train the meta model
    stacking_model.fit(base_predictions, data['y_train'], data['train_time_indices'][:len(data['y_train'])])
    
    # Use the meta model for final prediction
    test_base_predictions = {
        'informer': informer_preds,
        'lightgbm': lgbm_preds
    }
    
    ensemble_preds = stacking_model.predict(test_base_predictions)
    
    # Evaluate prediction results
    ensemble_metrics = evaluate_predictions(data['y_test'], ensemble_preds, data['target_scaler'])
    
    # Print evaluation metrics
    for metric, value in ensemble_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Inverse normalization
    y_test_original = data['target_scaler'].inverse_transform(data['y_test'].reshape(-1, 1)).reshape(data['y_test'].shape)
    ensemble_preds_original = data['target_scaler'].inverse_transform(ensemble_preds.reshape(-1, 1)).reshape(ensemble_preds.shape)
    
    # Plot using the date index of the test set
    plot_prediction(
        y_test_original, 
        ensemble_preds_original, 
        data['test_dates'][seq_len:seq_len+len(y_test_original)], 
        title='Load Forecast Result Comparison'
    )
    

if __name__ == "__main__":
    main() 