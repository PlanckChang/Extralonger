import torch.nn as nn
import torch

class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mode="temporal"):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mode = mode
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, adj_mask=None):
        batch_size = query.shape[0]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)

        attn_score = (query @ key) / self.head_dim**0.5 

        if self.mode == "spatial":
            global_attn_score = attn_score
            global_attn_score = torch.softmax(global_attn_score, dim=-1) 
            if type(adj_mask) == torch.Tensor:   
                local_attn_score = attn_score
                adj_mask = adj_mask.to(attn_score.device)      
                local_attn_score = attn_score.masked_fill(~adj_mask, -torch.inf)  # fill in-place
                local_attn_score  = torch.softmax(local_attn_score, dim=-1)
                global_attn_score = (global_attn_score + local_attn_score) / 2.0
            out = global_attn_score @ value  
        elif self.mode == "temporal":
            attn_score = torch.softmax(attn_score, dim=-1)
            out = attn_score @ value
        
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        out = self.out_proj(out)
        return out 

class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mode="temporal"
    ):
        super().__init__()
        self.mode = mode
        self.attn = AttentionLayer(model_dim, num_heads, mode)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, adj_mask=None):

        # x: (batch_size, time_step, model_dim) or (batch_size, num_nodes, model_dim)
        residual = x
        if self.mode == "temporal":
            out = self.attn(x, x, x)  
        elif self.mode == "spatial":
            out = self.attn(x, x, x, adj_mask=adj_mask)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        
        return out

class TransposeX(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class Extralonger(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=256,
        tod_embedding_dim=128,
        dow_embedding_dim=128,
        spatial_embedding_dim=256,
        feed_forward_dim=1024,
        num_heads=8,
        num_layers=1,
        adj_mask = None,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.noise_embedding_dim = input_embedding_dim

        self.model_dim = (
            input_embedding_dim 
            + tod_embedding_dim
            + dow_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.adj_mask = adj_mask
        
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
            
        self.input_noise = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes))
            )

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, mode="temporal")
                for _ in range(num_layers)
            ]
        )
        self.t_input  = nn.Sequential(
            nn.Linear(self.num_nodes, self.input_embedding_dim), 
        )    
        if in_steps != out_steps:
            self.t_output = nn.Sequential(
                nn.Linear(self.model_dim, self.num_nodes),
                TransposeX(1, 2),
                nn.Linear(self.in_steps, self.out_steps),
                TransposeX(1, 2),
            ) 
        else:          
            self.t_output = nn.Linear(self.model_dim, self.num_nodes)
                    
        self.s_input  = nn.Linear(self.in_steps, self.model_dim - self.spatial_embedding_dim)
        self.s_output = nn.Linear(self.model_dim, self.out_steps)
        
        self.attn_layers_s = nn.ModuleList(
            [  
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, mode="spatial")
                for _ in range(num_layers)
            ]
        )
        
        self.mix_proj = nn.Sequential(
            TransposeX(1, 2),
            nn.Linear(self.model_dim, self.num_nodes),
            TransposeX(1, 2),
            nn.Linear(self.in_steps, self.out_steps),
            TransposeX(1, 2)
        )
        
        self.attn_layers_mix_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, mode="temporal")
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_mix_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.in_steps, feed_forward_dim, 2, mode="spatial")
                for _ in range(num_layers)
            ]
        ) 
        
    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        
        if self.tod_embedding_dim > 0:
            tod = x[:, :, 0, 1].squeeze()
        if self.dow_embedding_dim > 0:
            dow = x[:, :, 0, 2].squeeze()
        x = x[:, :, :, 0].squeeze() 
        x = x + self.input_noise.expand(
            size=(batch_size, *self.input_noise.shape)
        )
        
        x_s = x
        x_t = x
     
        x_t = self.t_input(x_t)
        features = [x_t]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, dow_embedding_dim)
            features.append(dow_emb)   
        x_t = torch.cat(features, dim=-1)  # (batch_size, in_steps, model_dim)
        x_mix = x_t
        
    # temporal part
        for attn in self.attn_layers_t:
            x_t = attn(x_t)  
        x_t = self.t_output(x_t) # (batch_size, out_steps, num_nodes)
        
    # mixture part
        for attn in self.attn_layers_mix_t:
            x_mix = attn(x_mix)
        x_mix = x_mix.transpose(1, 2)     
        for attn in self.attn_layers_mix_s:
            x_mix = attn(x_mix)
        x_mix = self.mix_proj(x_mix)

    # spatial part
        x_s = self.s_input(x_s.transpose(1, 2))    # (batch_size, num_nodes, model_dim)
        features = [x_s]
        if self.spatial_embedding_dim > 0: 
            spatial_emb = self.node_emb.expand(
                batch_size, *self.node_emb.shape
            )
            features.append(spatial_emb)   
        x_s = torch.cat(features, dim=-1)  # (batch_size, num_nodes, model_dim)
        
        for attn in self.attn_layers_s:
            x_s = attn(x_s, adj_mask=self.adj_mask)
        x_s = self.s_output(x_s).transpose(1, 2) # (batch_size, out_steps, num_nodes)
        
        out = (2 * x_mix + x_s + x_t ) / 4.0
        out = out.view(batch_size, self.out_steps, self.num_nodes, self.output_dim)
        return out

if __name__ == "__main__":
    pass