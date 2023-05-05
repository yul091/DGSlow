from typing import *
import torch
import inspect

from transformers.generation_utils import BeamSearchScorer
from transformers import LogitsProcessorList, StoppingCriteriaList

def dialogue(
        module,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        exponential_decay_length_penalty: Optional[Tuple[int, float]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs):
    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else module.config.bos_token_id
    num_beams = num_beams if num_beams is not None else module.config.num_beams
    length_penalty = length_penalty if length_penalty is not None else module.config.length_penalty
    early_stopping = early_stopping if early_stopping is not None else module.config.early_stopping
    num_beam_groups = num_beam_groups if num_beam_groups is not None else module.config.num_beam_groups
    do_sample = do_sample if do_sample is not None else module.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else module.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else module.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else module.config.eos_token_id

    # output_scores = output_scores if output_scores is not None else module.config.output_scores
    output_scores = True
    output_attentions = output_attentions if output_attentions is not None else module.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else module.config.output_hidden_states
    )
    # return_dict_in_generate = (
    #     return_dict_in_generate if return_dict_in_generate is not None else module.config.return_dict_in_generate
    # )
    return_dict_in_generate = True

    if pad_token_id is None and eos_token_id is not None:
        # special case if pad_token_id is not defined
        # logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = module._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(module.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = module._prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    if module.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = module._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if module.config.is_encoder_decoder:
        input_ids = module._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
        )
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 5. Prepare `max_length` depending on other stopping criteria
    # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
    if max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids.shape[-1]
    elif max_length is not None and max_new_tokens is not None:
        pass
        # Both are set, this is odd, raise a warning
        # warnings.warn(
        #     "Both `max_length` and `max_new_tokens` have been set "
        #     f"but they serve the same purpose. `max_length` {max_length} "
        #     f"will take priority over `max_new_tokens` {max_new_tokens}.",
        #     UserWarning,
        # )
    # default to config if still None
    max_length = max_length if max_length is not None else module.config.max_length

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if module.config.is_encoder_decoder else "input_ids"
        # logger.warning(
        #     f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
        #     "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        # )

    # 6. determine generation mode
    is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
    is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
    is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)

    if num_beam_groups > num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    # # 7. prepare distribution pre_processing samplers
    # logits_processor = module._get_logits_processor(
    #     input_ids_seq_length=input_ids.shape[-1],
    #     exponential_decay_length_penalty=exponential_decay_length_penalty,
    #     renormalize_logits=renormalize_logits,
    #     repetition_penalty=repetition_penalty,
    #     no_repeat_ngram_size=no_repeat_ngram_size,
    #     encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
    #     encoder_input_ids=inputs_tensor,
    #     bad_words_ids=bad_words_ids,
    #     min_length=min_length,
    #     max_length=max_length,
    #     eos_token_id=eos_token_id,
    #     forced_bos_token_id=forced_bos_token_id,
    #     forced_eos_token_id=forced_eos_token_id,
    #     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #     num_beams=num_beams,
    #     num_beam_groups=num_beam_groups,
    #     diversity_penalty=diversity_penalty,
    #     remove_invalid_values=remove_invalid_values,
    #     logits_processor=logits_processor,
    # )

    # 8. prepare stopping criteria
    stopping_criteria = module._get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
    )

    # 9. go into different generation modes
    if is_greedy_gen_mode:
        if num_return_sequences > 1:
            raise ValueError(
                f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
            )

        # 10. run greedy search
        return module.greedy_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_sample_gen_mode:
        # 10. prepare logits warper
        logits_warper = module._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
        )

        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = module._expand_inputs_for_generation(
            input_ids,
            expand_size=num_return_sequences,
            is_encoder_decoder=module.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample
        return module.sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_beam_gen_mode:
        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        # 10. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=module.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )
        # 11. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = module._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=module.config.is_encoder_decoder, **model_kwargs
        )
        # 12. run beam search
        return module.beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_beam_sample_gen_mode:
        # 10. prepare logits warper
        logits_warper = module._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
        )

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size * num_return_sequences,
            num_beams=num_beams,
            device=module.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = module._expand_inputs_for_generation(
            input_ids,
            expand_size=num_beams * num_return_sequences,
            is_encoder_decoder=module.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run beam sample
        return module.beam_sample(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_group_beam_gen_mode:
        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if num_beams % num_beam_groups != 0:
            raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        # 10. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=stopping_criteria.max_length,
            device=module.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=num_beam_groups,
        )
        # 11. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = module._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=module.config.is_encoder_decoder, **model_kwargs
        )
        # 12. run beam search
        return module.group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )