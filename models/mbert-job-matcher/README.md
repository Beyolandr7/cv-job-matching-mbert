---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:180
- loss:CosineSimilarityLoss
base_model: google-bert/bert-base-multilingual-cased
widget:
- source_sentence: 'TABLE GAMES DEALER Summary Throughout the years, I''ve been trained
    to give only the best service and customer care. It is not only about doing the
    job right, but to make sure the customer is satisfied with their experience. I''ve
    progressed from a cashier, to a waitress, to my most recent employment as a blackjack
    dealer, giving me a total of three '
  sentences:
  - 'APPIAN BPM Architect Position: APPIAN BPM Architect Location: Washington DC (Hybrid)
    Experience: 13+ Years Work type: C2C & W2 PETADATA is currently looking to hire
    for the position of APPIAN BPM Architect for one of their clients. Roles & Responsibilities:
    The Ideal candidate should develop and test user management business process designs,
    in App'
  - 'Accountant (3+ years AP/AR exp. req.) Accountant (3+ years AP/AR exp. req.)Location:
    Cleveland, OHSalary: $55,000-$60,000 + BenefitsBenefits: Medical, Dental, Vision,
    Life, 401k + match, Holidays, Sick, and PTOJob Type: Full-Time, HybridTypical
    Hours: 8-5 M-F, 40 hoursStart Date: ASAPSponsorship is not available Accountant
    Description Our client is'
  - Lead Lab Engineer Description - External At Trane TechnologiesTM and through our
    businesses including Trane® and Thermo King®, we create innovative climate solutions
    for buildings, homes, and transportation that challenge what’s possible for a
    sustainable world. We're a team that dares to look at the world's challenges and
    see impactful possibiliti
- source_sentence: 'FIELD NURSE Professional Summary California-licensed Vocational
    Nurse with 25+ years of experience. Maintain strong reputation for achieving high
    levels of patient satisfaction. Works effectively as a team member and interact
    in a positive manner with patients, including difficult individuals. Core Qualifications
    CA License #VN133323 Skill Highligh'
  sentences:
  - 'Private Executive Chef (Michelin Star) On-site Position in Atherton, CAFull Time
    (40 hours/week), Non-Exempt/Hourly. Work Hours: 9:00 am - 5:30 pm (Weekend availability
    and flexibility required) COMPENSATION: $50 - $82/hour, DOE. (Non-Exempt/Hourly)
    Discretionary Bonus (Up to 20%) Health Insurance Paid Vacation, Sick Days Paid
    Meal Breaks Regular R'
  - Clinical Sales Specialist, Community Specialties - Yonkers, NY Clinical Sales
    Specialist, Community Specialties Yonkers, NY Developing innovative therapies
    is one of the most challenging, most essential, and personally rewarding fields
    in science. This is the most exciting time to be a part of Astellas, a company
    with a uniquely collaborative and p
  - 'QRM Application Support Engineer A financial firm is looking for a QRM Application
    Support Engineer to join their team in New York, NY. Compensation: $150-180k Responsibilities
    Install and configure all QRM applications Periodically validate the environment
    through diagnostic tests Liaise across IT domains to deliver QRM including but
    not limited t'
- source_sentence: INFORMATION TECHNOLOGY SPECIALIST I Professional Summary Skills
    account management, cables, cabling, Help Desk, Linux, MS Exchange server, Sharepoint,
    network security, desktop support, Unix, HP Unix, VBscript, Visual Basic Programming,
    Windows Server Active Directory account management, Help Desk, Linux, MS Exchange
    server 2010, network/server sec
  sentences:
  - 'Assistant Executive Chef / Kitchen Manager Hiring immediately for full-time Assistant
    Executive Chef or Kitchen Manager or Culinary Manager or Senior Kitchen Manager:
    Benefits We Offer Eligible for medical, dental, 401(K) retirement plan and paid
    time offCompetitive pay and bonus eligibilityPaid management training programOpportunity
    to work with a'
  - 'Medical Assistant Company Description Medically Modern is a leading provider
    of innovative Continuous Glucose Monitoring (CGM) solutions for individuals managing
    diabetes. Our mission is to democratize access to the latest in diabetes care,
    ensuring cutting-edge technology is within reach for all patients. We handle every
    step of the process, from '
  - 'Assistant Sales Manager Are you ready to grow your career? Empire Today is looking
    for an Assistant Sales Manager to support our dynamic Sales Team. The Assistant
    Sales Manager is responsible for assisting the market Sales Manager in the achievement
    of the Market sales goals by coaching and developing Sales Consultants and ensuring
    compliance . We '
- source_sentence: SOUS CHEF Summary To bring my positive attitude, strong work ethic
    and strong team attitude to a new career opportunity. Highlights Adaptive team
    player Customer service expert Strong organizational skills Deadline-driven Focused
    and driven Positive attitude Accomplishments Supported all kitchen operations
    when chef was absent. Assisted in maintain
  sentences:
  - Sr. Engineer, Failure Analysis & Cell Development ROLE QuantumScape is seeking
    an experienced and highly motivated failure analysis engineer. You will work in
    our state-of-the art lab, developing new test methods and characterization techniques
    that steer improvements to our novel solid-state battery processes and materials.
    REQUIRED EXPERIENCE & S
  - 'Restaurant Crewmember - Cook, Cashier, and Customer Service Starting hiring pay
    range (based on location, experience, qualifications, etc.): $12+ / hour Restaurant
    Crewmembers at Raising Cane’s will wear many hats (including a Raising Cane’s
    hat) while working hard and having fun as a critical part of the Restaurant team.
    We are hiring immediately '
  - Cook PRN Description Introduction Do you have the PRN career opportunities as
    a(an) Cook PRN you want with your current employer? We have an exciting opportunity
    for you to join Parkridge Medical Center which is part of the nation's leading
    provider of healthcare services, HCA Healthcare. Benefits Parkridge Medical Center,
    offers a total rewards pa
- source_sentence: SALES Professional Summary Experienced manager with excellent client
    and project management skills. Action-oriented with strong ability to communicate
    effectively with technology, executive, and business audiences. Analyst with extensive
    experience in Microsoft Office . Proficiencies include Microsoft Word and Microsoft
    Outlook . Can effectively mu
  sentences:
  - Lead Lab Engineer Description - External At Trane TechnologiesTM and through our
    businesses including Trane® and Thermo King®, we create innovative climate solutions
    for buildings, homes, and transportation that challenge what’s possible for a
    sustainable world. We're a team that dares to look at the world's challenges and
    see impactful possibiliti
  - Banquet Chef Description:ESSENTIAL FUNCTIONS:Responsible for writing transfers
    and food requisitions which are overseen by the Executive Chef or Executive Sous
    Chef.Prepare, together with assigned cooks, food items designated by the Executive
    Chef according to the property's standard recipes and policies.Promote teamwork
    between kitchen and banquet
  - Project Manager HRIS Job SummaryThe Project Manager - HRIS will be a primary figure
    in leading the day-to-day project planning, execution coordination and tracking,
    and facilitation of change (both technical and non-technical) across the company
    for major business system implementation projects new software implementations
    (Deltek, Procore or simil
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on google-bert/bert-base-multilingual-cased

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased). It maps inputs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) <!-- at revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'SALES Professional Summary Experienced manager with excellent client and project management skills. Action-oriented with strong ability to communicate effectively with technology, executive, and business audiences. Analyst with extensive experience in Microsoft Office . Proficiencies include Microsoft Word and Microsoft Outlook . Can effectively mu',
    'Project Manager HRIS Job SummaryThe Project Manager - HRIS will be a primary figure in leading the day-to-day project planning, execution coordination and tracking, and facilitation of change (both technical and non-technical) across the company for major business system implementation projects new software implementations (Deltek, Procore or simil',
    "Lead Lab Engineer Description - External At Trane TechnologiesTM and through our businesses including Trane® and Thermo King®, we create innovative climate solutions for buildings, homes, and transportation that challenge what’s possible for a sustainable world. We're a team that dares to look at the world's challenges and see impactful possibiliti",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6449, 0.4691],
#         [0.6449, 1.0000, 0.4954],
#         [0.4691, 0.4954, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 180 training samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence1                                                                          | sentence2                                                                           | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                              | float                                                          |
  | modality | text                                                                               | text                                                                                |                                                                |
  | details  | <ul><li>min: 59 tokens</li><li>mean: 79.14 tokens</li><li>max: 99 tokens</li></ul> | <ul><li>min: 64 tokens</li><li>mean: 82.33 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.57</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence1                                                                                                                                                                                                                                                                                                                                                                   | sentence2                                                                                                                                                                                                                                                                                                                                                                   | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>ASP.NET WEB DEVELOPER Accomplishments Won Associate of the Month award at CYIENT Member of the team which won the Team of the Month award for projects such as DEBTS and eRFA Experience April 2015 to Current Company Name City , State ASP.Net Web Developer Responsibilities: Migration of IU Print website (print.iu.edu) from ColdFusion to ASP.Net using</code> | <code>Assistant Facility Manager, Valet Operations Overview SP+ is driven by Our Promise, “Making Every Moment Matter for a World on the Go.” Our team understands that time is scarce, so we are intentional about putting ourselves in the shoes of those we serve. The result is a unique combination of talented people and industry-leading technology that ena</code> | <code>1.0</code> |
  | <code>AS INFORMATION TECHNOLOGY PROJECT MANAGER Summary SUMMARY Skilled IT professional with 7 years of proven success in developing and leading cross-functional technical teams to execute and deliver major technology initiatives using the Agile methodology and Scaled Agile Framework (SAFe). Well-versed in managing projects with co-located and off-site t</code> | <code>Case Developer At Morgan & Morgan, the work we do matters. For millions of Americans, we’re their last line of defense against insurance companies, large corporations or defective goods. From attorneys to client support staff, creative marketing to operations teams, every member of our firm has a key role to play in the winning fight for consumer r</code> | <code>1.0</code> |
  | <code>DATABASE PROGRAMMER/ANALYST (.NET DEVELOPER) Summary Software Engineer with 2 years in Web Developer specializing in front end development. Proficient in Software Development Life Cycle (SDLC) and SCRUM AGILE methodologies of development process to produce software solutions by team. Involved in designing, planning, developing and implementing Micr</code> | <code>Activities Assistant Position Details Shift: 10 a.m. - 6 p.m. Status: PRN Live the Mission PRN as needed Position Summary The Activities Assistant assists the Director of Activities in planning, organizing, developing, and directing quality activities for patients, ensuring that the recreational, physical, intellectual, spiritual, and social needs </code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss",
      "cos_score_transformation": "torch.nn.modules.linear.Identity"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `learning_rate`: 8e-05
- `warmup_steps`: 3

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 8e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 3
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4167 | 5    | 0.2843        |
| 0.8333 | 10   | 0.2510        |
| 1.25   | 15   | 0.2404        |
| 1.6667 | 20   | 0.2348        |


### Training Time
- **Training**: 2.3 minutes

### Framework Versions
- Python: 3.14.5
- Sentence Transformers: 6.0.0
- Transformers: 5.15.1
- PyTorch: 2.13.0+cpu
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->