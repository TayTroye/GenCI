from transformers import T5Config, T5ForConditionalGeneration
from collections import defaultdict,Counter
import random
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine, MLPLayers

import numpy as np
from torch.nn.init import xavier_normal_, constant_



class GenCI(ContextRecommender):
    """This is an extension of SASRec, which concatenates item representations and item attribute representations
    as the input to the model.
    """


    def __init__(self, config, dataset):
        super(GenCI, self).__init__(config, dataset)

        self.LABEL_FIELD = config["LABEL_FIELD"]

        # load parameters info

        self.embedding_size = config["embedding_size"] 
        self.dropout_prob = config["dropout_prob"]
        self.dropout_prob_fm = config["dropout_prob_fm"]

        self.device = config["device"]
        self.gamma = config["gamma"]
        self.delta = config["delta"]
        self.num_code_book = config['n_codebooks']
        self.book_dim = config["book_size"]

        self.filed2token = dataset.field2token_id

        self.user_id = dataset.field2id_token['user_id']
        self.item_id = dataset.field2id_token['item_seq']
        self.n_item = dataset.item_num

        self.item_seq_embed = config['item_seq_embed']
        self.num_heads_ca = config['num_heads_ca']
        self.croatt_dropout_prob = config['croatt_dropout_prob']


        self.token2code  = dataset.token2code
        self.code2token  = dataset.code2token
        fourth_col_codes = fourth_col_codes = set(c for c in self.token2code[:, 3].tolist() if c != 0)
        self.code2item_ids = defaultdict(set) 

        for code_tuple, item_id in self.code2token.items():
            for code in code_tuple:
                self.code2item_ids[code].add(item_id)  

        self.token2code = torch.tensor(self.token2code, dtype=torch.long, device=self.device)

        self.num_beams = config['num_beams']
        codebook_size = config['codebook_size']
        self.n_digit = dataset.n_digit
        n_codebook = [codebook_size] * self.n_digit
        base_user_token = sum(n_codebook) + 1
        n_user_token = config['n_user_tokens'] 
        eos_token = base_user_token + n_user_token 
        vocab_size = eos_token + 1
        padding_token = 0
        max_token_seq_len = config['max_item_seq_len'] * self.n_digit + 1 

        self.d_model = config['d_model']
        t5config = T5Config(
            num_layers=config['num_layers'], 
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate_t5'],
            activation_function=config['activation_function'],
            vocab_size= vocab_size,
            pad_token_id= padding_token,
            eos_token_id= eos_token, 
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions= max_token_seq_len,
        )

        self.t5 = T5ForConditionalGeneration(config=t5config)


        max_items_per_code = max(
            len(items) for code, items in self.code2item_ids.items()
            if code not in fourth_col_codes
        )
        self.max_seq_len = config['max_item_seq_length']



        # for accelerate beam search
        self.register_buffer(
            'user_preds_cache', 
            torch.zeros(dataset.user_num, self.num_code_book + 1, dtype=torch.long)
        )
        self.register_buffer(
            'user_encoder_cache',
            torch.zeros(dataset.user_num, self.max_seq_len * self.n_digit, self.d_model, dtype=torch.float)
        )
        self.register_buffer(
            'user_seq_output_cache',
            torch.zeros(dataset.user_num, self.d_model, dtype=torch.float)
        )

        self.cross_attn = nn.MultiheadAttention(
           self.d_model, self.num_heads_ca, dropout=self.croatt_dropout_prob,batch_first=True
            )

        if isinstance(self.token2code, torch.Tensor):
            first3_codes_all = self.token2code[:, :3].reshape(-1).tolist()
        else:
            first3_codes_all = np.array(self.token2code)[:, :3].reshape(-1).tolist()

        max_code_id = max(first3_codes_all)
        num_codes = max_code_id + 1


        first3_codes_set = set(first3_codes_all)

        max_items_per_code = max(
            len(items)
            for code, items in self.code2item_ids.items()
            if code in first3_codes_set  
        )

        code2items_tensor = torch.full(
            (num_codes, max_items_per_code),
            fill_value=0,  # PAD
            dtype=torch.long
        )


        for code, items in self.code2item_ids.items():
            if code in first3_codes_set:     
                item_list = list(items)[:max_items_per_code] 
                code2items_tensor[code, :len(item_list)] = torch.tensor(item_list, dtype=torch.long)

        self.code2items_tensor = code2items_tensor.to(self.device)
        self.allowed_masks = []
        allowed_codes_per_col = [
            set(
                c for c in torch.unique(self.token2code[:, i]).tolist()
                if c != 0 or i == 0
                # if c != 0
            )
            for i in range(self.token2code.shape[1])
        ]

        for col_codes in allowed_codes_per_col:
            mask = torch.full((vocab_size,), float('-inf'), device=self.device)
            mask[list(col_codes)] = 0.0 
            self.allowed_masks.append(mask)

                 
        
        field_idx = self.token_field_names.index('item_id')
        self.offset = self.token_embedding_table.offsets[field_idx]

        
        self.length = self.token_field_dims[field_idx]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.mlp_hidden_size_fm = config['mlp_hidden_size_fm']

        self.initializer_range = config["initializer_range"]

        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [
            self.embedding_size * (self.total_feature_num  )
        ] + self.mlp_hidden_size_fm
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob_fm)
        self.deep_predict_layer = nn.Linear(
            self.mlp_hidden_size_fm[-1], 1
        )  # Linear product to the final score


        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.item_seq_embedding = nn.Embedding(self.length,self.d_model , padding_idx=0)

        self.dnn_list = [
           (self.total_feature_num -1) * self.embedding_size + 2 * self.d_model 
        ] + self.mlp_hidden_size

        self.dnn_mlp_layers = MLPLayers(
            self.dnn_list, dropout=self.dropout_prob
        )
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.apply(self._init_weights)
        self.item_seq_embedding.weight.data[0].zero_()


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)



    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def generate(self, batch: dict, n_return_sequences: int = 1):
        """
        Generates sequences using beam search algorithm.

        Args:
            batch (dict): A dictionary containing input_ids and attention_mask.
            n_return_sequences (int): The number of sequences to generate.

        Returns:
            torch.Tensor: The generated sequences.
        """
        n_digit = self.n_digit
        outputs = self.beam_search(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=n_digit + 1,
            num_beams= self.num_beams,
            num_return_sequences=n_return_sequences,
            return_score=True
        )
        preds, scores = outputs
        preds = preds[:, 1:1 + n_digit].reshape(-1, n_return_sequences, n_digit)
        scores = scores.reshape(-1, n_return_sequences)
        return preds, scores

    def beam_search(
        self,
        input_ids,
        attention_mask,
        max_length=6,
        num_beams=1,
        num_return_sequences=1,
        return_score=False
    ):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Perform beam search to generate sequences using the specified model. 

        *** This implementation does not include stopping conditions based on end-of-sequence (EOS) tokens. Instead, the
        sequence generation is controlled solely by the `max_length` parameter. ***

        Note: In scenarios where the generation should explicitly detect and respond to EOS tokens 
        to terminate the sequence early, this function would need modifications. In the current setup,
        setting `max_length` to a suitable fixed value (e.g., 6) can serve the purpose by limiting
        the maximum sequence length.

        Parameters:
        - input_ids (torch.Tensor): Tensor of input ids.
        - attention_mask (torch.Tensor): Tensor representing the attention mask.
        - max_length (int): Maximum length of the sequence to be generated; controls when to stop extending the sequence.
        - num_beams (int): Number of beams for beam search.
        - num_return_sequences (int): Number of sequences to return.
        - return_score (bool): If True, returns a tuple of (sequences, scores) where 'scores' are the average log likelihood of the returned sequences.

        Returns:
        - torch.Tensor: The final decoder input ids from the beam search, or a tuple of (decoder_input_ids, scores) if 'return_score' is True.

        Example usage:
        # Assuming the model, input_ids, and attention_mask are predefined:
        sequences = beam_search(model, input_ids, attention_mask, max_length=6, num_beams=5, num_return_sequences=5)
        """

        batch_size = input_ids.shape[0]

        # Prepare beam search inputs
        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )
        # Store encoder_outputs to prevent running full forward path repeatedly
        with torch.no_grad():
            encoder_outputs = self.t5.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

        # Beam search loop
        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.t5(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )

            decoder_input_ids, beam_scores = self.beam_search_step(
                outputs.logits,
                decoder_input_ids,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams
            )

        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], \
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)

        return decoder_input_ids[selection_mask.view(-1), :]

    def prepare_beam_search_inputs(self, input_ids, attention_mask, batch_size, num_beams):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Prepares and duplicates the input data for beam search decoding.

        This function initializes decoder input IDs and beam scores, creates an offset for beam indices, 
        and expands the input_ids and attention_mask tensors to accommodate the specified number of beams for each instance in the batch.

        Parameters:
        - input_ids (torch.Tensor): The input IDs tensor of shape (batch_size, sequence_length) used for the encoder part of the model.
        - attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, sequence_length) indicating to the model which tokens should be attended to.
        - batch_size (int): The number of instances per batch in the input data.
        - num_beams (int): The number of beams to use in beam search. This expands the input data and scores accordingly.

        Returns:
        - input_ids (torch.Tensor): The expanded input IDs tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - attention_mask (torch.Tensor): The expanded attention mask tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - initial_decoder_input_ids (torch.Tensor): The initialized decoder input IDs for each beam, shape (batch_size * num_beams, 1).
        - initial_beam_scores (torch.Tensor): The initialized scores for each beam, flattened to a single dimension, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): An offset for each beam index to assist in reordering beams during the search, shape (batch_size * num_beams,).

        Each input sequence is replicated 'num_beams' times to provide separate candidate paths in beam search. Beam scores are initialized with 0 for the first beam and a very low number (-1e9) for others to ensure the first token of each sequence is chosen from the first beam.
        """

        decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=self.t5.device, dtype=torch.long)
        initial_decoder_input_ids = decoder_input_ids * self.t5.config.decoder_start_token_id

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9  # Set a low score for all but the first beam to ensure the first beam is selected initially
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))

        beam_idx_offset = torch.arange(batch_size, device=self.t5.device).repeat_interleave(num_beams) * num_beams

        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset


    def beam_search_step(self, logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Executes one step of beam search, calculating the next set of input IDs based on logits from a model.

        This function expands the current beam, calculates scores for all possible next tokens, selects the top tokens for each beam, and prepares the input IDs for the next iteration of the model. It utilizes logits output by the model to determine the most likely next tokens and updates the beam scores.

        Parameters:
        - logits (torch.Tensor): Logits returned from the model, shape (batch_size * num_beams, sequence_length, vocab_size).
        - decoder_input_ids (torch.Tensor): Current decoder input IDs, shape (batch_size * num_beams, current_sequence_length).
        - beam_scores (torch.Tensor): Current scores for each beam, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): Index offsets for each beam to handle batches correctly, shape (batch_size * num_beams,).
        - batch_size (int): Number of sequences being processed in a batch.
        - num_beams (int): Number of beams used in the beam search.

        Returns:
        - decoder_input_ids (torch.Tensor): Updated decoder input IDs after adding the next tokens, shape (batch_size * num_beams, current_sequence_length + 1).
        - beam_scores (torch.Tensor): Updated scores for each beam, shape (batch_size * num_beams,).

        The function selects the top `2 * num_beams` tokens from the logits based on their scores, reshapes and adjusts them based on the existing beam scores, and determines the next tokens to add to each beam path. The updated paths are then returned for use in the next iteration of the beam search.
        """
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]


        current_col = decoder_input_ids.shape[1] - 1  
        current_mask = self.allowed_masks[current_col]  
        next_token_logits = next_token_logits + current_mask   



        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)  # Calculate log softmax over the last dimension

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        # beam_idx_offset: beam_idx contains sequence indicies relative to each individual batch. We need to offset the indicies to retrieve the correct sequence in the corresponding batch
        # for example, when batch_size = 2, beam_size = 3, beam_idx_offset = [0, 0, 0, 3, 3, 3]
        decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        return decoder_input_ids, beam_scores




    def inference_accelerate(self, interaction,item_seq, next_items,item_seq_code, next_items_code):

        user_ids = interaction['user_id']
        B, L = item_seq.size()
        device = item_seq.device

        if self.training:
            input_ids = item_seq_code.view(B, -1)
            attention_mask = (input_ids != 0).long()
            labels = next_items_code.squeeze(1)


            t5_outputs = self.t5(input_ids=input_ids, 
                                 attention_mask=attention_mask,
                                 labels=labels)
            
            t5_loss = t5_outputs.loss
            encoder_representation = t5_outputs.encoder_last_hidden_state
            
            # 2. Get the final sequence output
            sequence_lengths = torch.sum(attention_mask, dim=1)
            last_token_indices = (sequence_lengths - 1).clamp(min=0)
            batch_indices = torch.arange(B, device=device)
            seq_output = encoder_representation[batch_indices, last_token_indices]
            
            # 3. Run the expensive generate pass
            batch_gen = {"input_ids": input_ids, "attention_mask": attention_mask}
            preds, _ = self.generate(batch_gen, 1)
            preds_flat = preds.squeeze(1)

            # 4. Update all caches
            self.user_preds_cache[user_ids] = preds_flat
            self.user_encoder_cache[user_ids] = encoder_representation
            self.user_seq_output_cache[user_ids] = seq_output
        else: 
            t5_loss = torch.tensor(0.0, device=device)  
            preds_flat = self.user_preds_cache[user_ids]
            encoder_representation = self.user_encoder_cache[user_ids]
            seq_output = self.user_seq_output_cache[user_ids]
            # We still need attention_mask for downstream logic if any
            input_ids = item_seq_code.view(B, -1)
            attention_mask = (input_ids != 0).long()

        first3_codes = preds_flat[:, :3]
        items_for_codes = self.code2items_tensor[first3_codes]
        items_emb = self.item_seq_embedding(items_for_codes)
        batch_cluster_embs = items_emb.mean(dim=2)
        next_items_embedding = self.item_seq_embedding(next_items).unsqueeze(1)
        

        attn_output, _ = self.cross_attn(query=batch_cluster_embs, key=encoder_representation, value=encoder_representation)
        across_out, _ = self.cross_attn(query=next_items_embedding, key=attn_output, value=attn_output)

        sparse_embedding, _ = self.embed_input_fields_ui(interaction)
        user_emb = sparse_embedding['user'].view(B, -1)
        item_emb = sparse_embedding['item'].view(B, -1)
        
        din_in = torch.cat((seq_output, across_out.squeeze(1), item_emb, user_emb), dim=-1)
        dnn_out = self.dnn_predict_layers(
            self.dnn_mlp_layers(din_in)
        )

        if self.training:
            # --- Calculate DeepFM for loss ---
            deepfm_all_embeddings = self.concat_embed_input_fields(
                interaction
            )
            batch_size = deepfm_all_embeddings.shape[0]
            y_fm = self.first_order_linear(interaction) + self.fm(deepfm_all_embeddings)
            y_deep = self.deep_predict_layer(
                self.mlp_layers(deepfm_all_embeddings.view(batch_size, -1))
            )
            y_df = y_fm + y_deep
        else:
            batch_size = dnn_out.size(0) # 
            y_df = torch.zeros(batch_size, 1, device=dnn_out.device)
 
        y_pred = dnn_out 

        return_dict = { "pred_output": y_pred.squeeze(-1),"t5_loss":t5_loss,"seq_output":seq_output,"fm_output":y_df.squeeze(-1),"sas_output":dnn_out.squeeze(-1)}
        
        return return_dict  # [B H]



    def forward(self, interaction,item_seq, next_items,item_seq_code, next_items_code):

        item_seq = interaction['item_seq']

        B, L = item_seq.size()


        input_ids = item_seq_code.view(B, -1)
        attention_mask = (input_ids != 0).long()
        labels = next_items_code.squeeze(1)

        t5_outputs = self.t5(input_ids = input_ids, 
                          attention_mask = attention_mask,
                          labels = labels) 

        t5_loss = t5_outputs.loss

        batch = {}
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        preds, scores = self.generate( batch , 1) 
        preds_flat = preds.squeeze(1)  
        first3_codes = preds_flat[:, :3] 
        items_for_codes = self.code2items_tensor[first3_codes]  # [B,3,max_items_per_code]
        items_emb = self.item_seq_embedding(items_for_codes)  
        batch_cluster_embs  = items_emb.mean(dim=2)  

        next_items_embedding = self.item_seq_embedding(next_items).unsqueeze(1)  
        encoder_representation = t5_outputs.encoder_last_hidden_state 
        
        sequence_lengths = torch.sum(attention_mask, dim=1)
        last_token_indices = (sequence_lengths - 1).clamp(min=0)
        batch_indices = torch.arange(encoder_representation.size(0), device=input_ids.device)


        attn_output, _ = self.cross_attn(query=batch_cluster_embs, key=encoder_representation, value=encoder_representation)
        across_out, _ = self.cross_attn(query=next_items_embedding, key=attn_output, value=attn_output)

        seq_output = encoder_representation[batch_indices, last_token_indices]

        sparse_embedding, _  = self.embed_input_fields_ui(interaction)
        user_emb = sparse_embedding['user'].view(B,-1)  
        item_emb = sparse_embedding['item'].view(B,-1)  
 
        din_in = torch.cat((seq_output , across_out.squeeze(1), item_emb, user_emb ),dim=-1)
        dnn_out = self.dnn_predict_layers(
            self.dnn_mlp_layers(din_in)
        )
        y_pred = dnn_out 


        if self.training:
            # --- Calculate DeepFM for loss ---
            deepfm_all_embeddings = self.concat_embed_input_fields(
                interaction
            )
            batch_size = deepfm_all_embeddings.shape[0]
            y_fm = self.first_order_linear(interaction) + self.fm(deepfm_all_embeddings)
            y_deep = self.deep_predict_layer(
                self.mlp_layers(deepfm_all_embeddings.view(batch_size, -1))
            )
            y_df = y_fm + y_deep
        else:
            batch_size = dnn_out.size(0) 
            y_df = torch.zeros(batch_size, 1, device=dnn_out.device)
 
        return_dict = { "pred_output": y_pred.squeeze(-1),"t5_loss":t5_loss,"seq_output":seq_output,"fm_output":y_df.squeeze(-1)}

        return return_dict 



    def calculate_loss(self, interaction):

        item_seq = interaction['item_seq']
        next_items = interaction['item_id']

        label = interaction[self.LABEL_FIELD]
        item_seq_code = self.token2code[item_seq] 
        next_items_code = self.token2code[next_items]

        return_dict = self.forward(interaction,item_seq, next_items,item_seq_code,next_items_code)

        pred_output = return_dict["pred_output"]

        t5_loss = return_dict['t5_loss']
        fm_output = return_dict['fm_output']

        ctr_loss = self.loss(pred_output, label)
        tri_loss = self.loss(pred_output,self.sigmoid(fm_output)) + self.loss(fm_output,label)
        loss =   ctr_loss  + self.gamma * t5_loss  + self.delta * tri_loss 

        return loss

    def predict(self, interaction):
        item_seq = interaction['item_seq']
        next_items = interaction['item_id']
        item_seq_code = self.token2code[item_seq] 
        next_items_code = self.token2code[next_items]

        return_dict = self.forward(interaction,item_seq, next_items,item_seq_code,next_items_code)

        pred_output = return_dict["pred_output"]        

        predition = self.sigmoid(pred_output)
        return predition


