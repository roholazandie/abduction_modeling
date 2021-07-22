from config import ModelConfig, DataConfig
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import load_dataset, get_input_embeds, get_text_from_logits


def backward_pass(model_config,
                  backward_model,
                  backward_tokenizer,
                  o2_logits
                  ):
    past = None
    last_embeds = None
    logits_so_far = None
    logits_so_far_complete = None

    for i in range(model_config.max_length):
        # Run model forward to obtain unperturbed logits
        if past is None:
            o1_embeds = get_input_embeds(backward_model.get_input_embeddings(), o2_logits, device=model_config.device)
            last_embeds = o1_embeds[:, -1:, :]

            if o2_logits.shape[1] > 1:
                model_output = backward_model(inputs_embeds=o1_embeds[:, :-1, :])
                past = model_output.past_key_values

        model_output = backward_model(past_key_values=past, inputs_embeds=last_embeds)
        backward_logit = model_output.logits
        past = model_output.past_key_values

        backward_logit = backward_logit[:, -1, :] / model_config.temperature_forward

        if i < model_config.length:
            # Mix backward and forward logits
            # todo this is probably h_logtis[:, length-i, :]
            # pert_logits = model_config.mix_rate * unpert_logits# + (1 - model_config.mix_rate) * h_logits[:, i,:]
            pert_logits = backward_logit  # + (1 - model_config.mix_rate) * h_logits[:, i,:]
        else:
            # Continue to complete the text
            pert_logits = backward_logit

        pert_logits = pert_logits.unsqueeze(1)
        if i < model_config.length:
            logits_so_far = pert_logits if logits_so_far is None else torch.cat((logits_so_far, pert_logits), dim=1)
        logits_so_far_complete = pert_logits if logits_so_far_complete is None else torch.cat(
            (logits_so_far_complete, pert_logits), dim=1)

        # Use a small temperature (0.1) so that the soft token representation is sharper,
        # and closer to a one-hot representation
        last_embeds = get_input_embeds(backward_model.get_input_embeddings(), pert_logits / 0.1,
                                       device=model_config.device)

    # Sample a text
    backward_text = get_text_from_logits(logits_so_far_complete, backward_tokenizer, temperature=1.0,
                                         top_k=model_config.top_k)

    print("backward generated:", " ".join(reversed(backward_text.split())))
    return logits_so_far


def forward_pass(model_config,
                 forward_model,
                 forward_tokenizer,
                 backward_logits,
                 o1_logits,
                 o1
                 ):
    past = None
    last_embeds = None
    logits_so_far = None
    logits_so_far_complete = None

    for i in range(model_config.max_length):
        # Run model forward to obtain unperturbed logits
        if past is None:
            o1_embeds = get_input_embeds(forward_model.get_input_embeddings(), o1_logits, device=model_config.device)
            last_embeds = o1_embeds[:, -1:, :]

            if o1_logits.shape[1] > 1:
                model_output = forward_model(inputs_embeds=o1_embeds[:, :-1, :])
                past = model_output.past_key_values

        model_output = forward_model(past_key_values=past, inputs_embeds=last_embeds)
        unpert_logits = model_output.logits
        past = model_output.past_key_values

        unpert_logits = unpert_logits[:, -1, :] / model_config.temperature_forward

        if i < model_config.length:
            # Mix backward and forward logits
            # todo this is probably h_logtis[:, length-i, :]
            pert_logits = model_config.mix_rate * unpert_logits + (1 - model_config.mix_rate) * backward_logits[:, i, :]
            # pert_logits = unpert_logits
        else:
            # Continue to complete the text
            pert_logits = unpert_logits

        pert_logits = pert_logits.unsqueeze(1)
        if i < model_config.length:
            logits_so_far = pert_logits if logits_so_far is None else torch.cat((logits_so_far, pert_logits), dim=1)
        logits_so_far_complete = pert_logits if logits_so_far_complete is None else torch.cat(
            (logits_so_far_complete, pert_logits), dim=1)

        # Use a small temperature (0.1) so that the soft token representation is sharper,
        # and closer to a one-hot representation
        last_embeds = get_input_embeds(forward_model.get_input_embeddings(), pert_logits / 0.1,
                                       device=model_config.device)

    # Sample a text
    forward_text = get_text_from_logits(logits_so_far_complete, forward_tokenizer, temperature=1.0,
                                        top_k=model_config.top_k)

    return logits_so_far, forward_text


def generate_abductive_explanation(model_config,
                                   forward_model,
                                   forward_tokenizer,
                                   backward_model,
                                   backward_tokenizer,
                                   o1_text,
                                   o2_text):
    o1 = forward_tokenizer(forward_tokenizer.bos_token + o1_text, return_tensors="pt").to(model_config.device)
    output_so_far = o1['input_ids']

    o2_text_reversed = ' '.join(list(reversed(o2_text.split())))
    o2 = backward_tokenizer(forward_tokenizer.eos_token + o2_text_reversed, return_tensors="pt").to(model_config.device)

    # use a very small temperature to mimic one-hot after softmax
    o1_logits = F.one_hot(output_so_far) / 0.00001
    o2_logits = F.one_hot(o2["input_ids"]) / 0.00001

    # ## The initialization pass to initialize the generation (its logits)
    # past = None
    # last_embeds = None
    # logits_so_far = None
    #
    # for i in range(model_config.length):
    #     # run model forward to obtain unperturbed logits
    #     if past is None and output_so_far is not None:
    #         last = output_so_far[:, -1:]
    #         last_embeds = forward_model.get_input_embeddings()(last)
    #
    #         if output_so_far.shape[1] > 1:
    #             model_output = forward_model(output_so_far[:, :-1])
    #             past = model_output.past_key_values
    #
    #     model_output = forward_model(past_key_values=past, inputs_embeds=last_embeds)
    #     unpert_logits = model_output.logits
    #     past = model_output.past_key_values
    #
    #     unpert_logits = unpert_logits[:, -1, :] / model_config.temperature_forward
    #
    #     unpert_logits = unpert_logits.unsqueeze(1)
    #
    #     logits_so_far = unpert_logits if logits_so_far is None else torch.cat((logits_so_far, unpert_logits), dim=1)
    #
    #     last_embeds = get_input_embeds(forward_model.get_input_embeddings(), unpert_logits / 0.01,
    #                                    device=model_config.device)

    backward_logits = backward_pass(model_config,
                                    backward_model,
                                    backward_tokenizer,
                                    o2_logits
                                    )

    logits_so_far, forward_text = forward_pass(model_config,
                                               forward_model,
                                               forward_tokenizer,
                                               backward_logits=backward_logits,
                                               o1_logits=o1_logits,
                                               o1=o1)

    return forward_text


if __name__ == '__main__':
    model_config = ModelConfig.from_json("configs/model_config.json")
    data_config = DataConfig.from_json("configs/data_config.json")

    forward_model = GPT2LMHeadModel.from_pretrained(model_config.forward_pretrained_model)
    forward_model.to(model_config.device)
    forward_model.eval()
    forward_tokenizer = GPT2Tokenizer.from_pretrained(model_config.forward_pretrained_model)

    # Freeze GPT-2 weights
    for param in forward_model.parameters():
        param.requires_grad = False

    backward_model = GPT2LMHeadModel.from_pretrained(model_config.backward_pretrained_model)
    backward_model.to(model_config.device)
    backward_model.eval()
    backward_tokenizer = GPT2Tokenizer.from_pretrained(model_config.backward_pretrained_model)

    # Freeze GPT-2 weights
    for param in backward_model.parameters():
        param.requires_grad = False

    records = load_dataset(data_config.abductive_dataset)
    print(records)

    for r in records:
        o1_text = '<|endoftext|>'.join([r['obs2'], r['obs1']])
        o2_text = r['obs2']

        text = generate_abductive_explanation(model_config,
                                       forward_model,
                                       forward_tokenizer,
                                       backward_model,
                                       backward_tokenizer,
                                       o1_text,
                                       o2_text)
        print("final", text)