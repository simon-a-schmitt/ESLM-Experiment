from transformers import (
    AutoModel,
    T5EncoderModel,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ESLM model without KGE
class ESLM(nn.Module):
    """
    Employing contextual language models for entity summarization tasks
    """
    def __init__(self, model_name, model_base, mode, mlp_hidden_dim=512):
        """
        Model initialization
        
        Attributes:
            - model_name (str): the name of contextual language model in lower case 
                (e.g., 't5', 'bert', 'ernie').
            - model_base (str): the parameter specifies the exact pre-trained model variant to be loaded 
                (e.g., 't5-base', 'bert-base-uncased') related to the model name.
            - mlp_hidden_dim (int): the size of the hidden layers within the Multi-Layer Perceptron (MLP) part of the model.
        """
        super(ESLM, self).__init__()
        self.model_name = model_name
        self.model_base = model_base

        # Load lm encoder
        if self.model_name=="t5":
            self.lm_encoder = T5EncoderModel.from_pretrained(model_base)
            self.feat_dim = self.lm_encoder.config.d_model
            
        else:
            self.lm_encoder = AutoModel.from_pretrained(model_base)
            self.feat_dim = list(self.lm_encoder.modules())[-2].out_features

        # Define second-level encoder and projection layers (for second-level encoder approach only)
        if mode == "second-level-encoder":
            self.second_level_encoder = T5EncoderModel.from_pretrained(model_base)

            self.projection_layer = nn.Linear(self.feat_dim, self.feat_dim)

            self.projection_layer_2 = nn.Linear( self.feat_dim, self.feat_dim)
            
        # Define attention layer
        self.attention = nn.Linear(self.feat_dim, 1)
        self.regression = nn.Linear(self.feat_dim, 1)  # Output layer for regression
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # Output layer for regression
        )

        # Selected mode (Standard, Modified-Information-Fusion, Prompting, Second-level Encoder)
        self.mode = mode


    # Forward pass
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Attributes:
            - input_ids (tensor): sequences of integers representing tokens mapped from a vocabulary
            - attention_mask (tensor): the tensors comprise of 1 and 0 to helps the model to distinguish 
                between meaningful data and padding data
        """

        # Embedding of tokens by llm
        encoder_output = self.lm_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


        if self.mode == "second-level-encoder":

            # Average over the sequence
            pooled_output = encoder_output.mean(dim=1)

            # Project to the same dimension as the KG embeddings
            pooled_output = self.projection_layer(pooled_output)
        
            # Combine triples into a batch of size 1 for second encoder
            second_input_ids = pooled_output.unsqueeze(0)  # Shape: (1, num_triples, hidden_dim)
            second_attention_mask = torch.ones(second_input_ids.size()[:-1], device=input_ids.device)
            
            # Pass through second-level encoder
            second_encoder_output = self.second_level_encoder(
                   inputs_embeds=pooled_output.unsqueeze(0),
                   attention_mask=second_attention_mask,
            ).last_hidden_state

            # Remove batch dimension for subsequent processing
            pooled_output = second_encoder_output.squeeze(0)  # Shape: (num_triples, hidden_dim)

            # Project back to the original dimension
            pooled_output = self.projection_layer_2(pooled_output)

        elif self.mode == "modified-information-fusion":
            # pooled_output = self.projection_layer(encoder_output)

            # Mask padding tokens (attention_mask == 0) and compute mean only over non-padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(encoder_output)  # Shape: (num_triples, seq_len, embedding_dim)
            sum_embeddings = (encoder_output * mask).sum(dim=1)  # Sum of embeddings for non-padded tokens
            count_non_padding = mask.sum(dim=1)  # Count of non-padded tokens

            pooled_output = sum_embeddings / count_non_padding

        elif self.mode == "prompting":

            # Average over the sequence
            pooled_output = encoder_output.mean(dim=1)

        elif self.mode == "standard":

            # Average over the sequence
            pooled_output = encoder_output.mean(dim=1)

        else:
            raise ValueError("Invalid mode")
        
        # Apply attention mechanism
        attn_weights = F.softmax(self.attention(pooled_output), dim=-1)
        combined_output = attn_weights * pooled_output
        
        # Pass through MLP
        regression_output = self.mlp(combined_output)

        # Apply activation 
        # For outputs bounded between 0 and 1
        regression_output = F.softmax(regression_output, dim=0) # use dim=0 due to not implemented data batches

        return regression_output


# ESLM model with KGE
class ESLMKGE(nn.Module):
    """
    Implementation ESLM Enrichment by augmenting Knowledge Graph Embeddings(KGEs)
    """
    def __init__(self, model_name, model_base, mode, kg_embedding_dim=1200, mlp_hidden_dim=512):
        """
        Model initialization
        
        Attributes:
            - model_name (str): the name of contextual language model in lower case 
                (e.g., 't5', 'bert', 'ernie').
            - model_base (str): the parameter specifies the exact pre-trained model variant to be loaded 
                (e.g., 't5-base', 'bert-base-uncased') related to the model name.
            - kg_embedding_dim (int): the size of the embeddings used for the knowledge graph components
            - mlp_hidden_dim (int): the size of the hidden layers within the Multi-Layer Perceptron (MLP) part of the model.
        """
        super(ESLMKGE, self).__init__()

        # Load lm encoder
        if model_name == "t5":
            # Only encoder part is used
            self.lm_encoder = T5EncoderModel.from_pretrained(model_base)
            self.feat_dim = self.lm_encoder.config.d_model
            self.num_heads = self.lm_encoder.config.num_heads   
            self.num_layers = self.lm_encoder.config.num_layers
        else:
            self.lm_encoder = AutoModel.from_pretrained(model_base)
            self.feat_dim = list(self.lm_encoder.modules())[-2].out_features
            self.num_heads = self.lm_encoder.config.num_attention_heads
            self.num_layers = self.lm_encoder.config.num_hidden_layers

        # Define second-level encoder and projection layers (for second-level encoder approach only)
        if mode == "second-level-encoder":
            self.second_level_encoder = T5EncoderModel.from_pretrained(model_base)

            self.projection_layer = nn.Linear(self.feat_dim + kg_embedding_dim, self.feat_dim)

            self.projection_layer_2 = nn.Linear( self.feat_dim, self.feat_dim + kg_embedding_dim)

        
        # Attention layer
        self.attention = nn.Linear(self.feat_dim + kg_embedding_dim, 1)
        
        # Refression layer
        self.regression = nn.Linear(self.feat_dim + kg_embedding_dim, 1)  # Output layer for regression
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim + kg_embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # Output layer for regression
        )

        # Selected mode (Standard, Modified-Information-Fusion, Prompting, Second-level Encoder)
        self.mode = mode


    # Forward pass
    def forward(self, input_ids, attention_mask, kg_embeddings):
        """
        Forward pass
        
        Attributes:
            - input_ids (tensor): sequences of integers representing tokens mapped from a vocabulary
            - attention_mask (tensor): the tensors comprise of 1 and 0 to helps the model to distinguish 
                between meaningful data and padding data
            - kg_embeddings (tensor): the tensors represent KGEs, where each embedding is likely 
                a vectorized representation of a triple (e.g., subject, predicate, object) from 
                a knowledge graph
        """

        # Embedding of tokens by llm
        # In: (num_triples, seq_len)
        # Result: (num_triples, seq_len, hidden_dim)
        encoder_output = self.lm_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Expand KG Embeddings
        # Before: (num_triples, 1, 1200)
        # After: (num_triples, seq_len, 1200)
        kg_embeddings_expanded = kg_embeddings.expand(-1, encoder_output.size(1), -1)
           

        # Combine with lm encoder output
        combined_embeddings = torch.cat([encoder_output, kg_embeddings_expanded], dim=-1)


        if self.mode == "second-level-encoder":

            # Average over the sequence
            pooled_output = combined_embeddings.mean(dim=1)

            # Project to the same dimension as the KG embeddings
            pooled_output = self.projection_layer(pooled_output)
        
            # Combine triples into a batch of size 1 for second encoder
            second_input_ids = pooled_output.unsqueeze(0)  # Shape: (1, num_triples, hidden_dim)
            second_attention_mask = torch.ones(second_input_ids.size()[:-1], device=input_ids.device)
            
            # Pass through second-level encoder
            second_encoder_output = self.second_level_encoder(
                   inputs_embeds=pooled_output.unsqueeze(0),
                   attention_mask=second_attention_mask,
            ).last_hidden_state

            # Remove batch dimension for subsequent processing
            pooled_output = second_encoder_output.squeeze(0)  # Shape: (num_triples, hidden_dim)

            # Project back to the original dimension
            pooled_output = self.projection_layer_2(pooled_output)

        elif self.mode == "modified-information-fusion":
            # pooled_output = self.projection_layer(combined_embeddings)

            # Mask padding tokens (attention_mask == 0) and compute mean only over non-padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(combined_embeddings)  # Shape: (num_triples, seq_len, embedding_dim)
            sum_embeddings = (combined_embeddings * mask).sum(dim=1)  # Sum of embeddings for non-padded tokens
            count_non_padding = mask.sum(dim=1)  # Count of non-padded tokens

            pooled_output = sum_embeddings / count_non_padding

        elif self.mode == "prompting":

            # Average over the sequence
            pooled_output = combined_embeddings.mean(dim=1)

        elif self.mode == "standard":

            # Average over the sequence
            pooled_output = combined_embeddings.mean(dim=1)

        else:
            raise ValueError("Invalid mode")

        # Apply attention mechanism
        # (num_triples, 1)
        attn_weights = F.softmax(self.attention(pooled_output), dim=-1)
        # Broadcasting for attn_weights
        

        # Result: (num_triples, 1200 + hidden_dim)
        combined_output = attn_weights * pooled_output
        
        # Pass through MLP
        # Result: (num_triples, 1)
        regression_output = self.mlp(combined_output)
        
        # Apply activation 
        # For outputs bounded between 0 and 1
        # Result: (num_triples, 1)
        regression_output = F.softmax(regression_output, dim=0) # use dim=0 due to not implemented data batches

        return regression_output
