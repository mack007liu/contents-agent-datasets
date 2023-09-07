New developments after I post the above on Zulip on Feb 14:

LLaMA/Alpaca ecosystem
LLaMA model leak, torrent, Hugging Face, hash
Facebook is going after LLaMA repos with DMCA's, https://github.com/github/dmca/blob/master/2023/03/2023-03-21-meta.md

Most renowned: StableVicuna, Vicuna, GPT4All

Quantization, C++ implementation, CPU inference: https://github.com/ggerganov/llama.cpp, https://github.com/antimatter15/alpaca.cpp

Alpaca-LoRA, https://github.com/tloen/alpaca-lora, 4bit (instruction fine-tuning)

Multimodal: LLaVA, MiniGPT-4. Compare Flamingo, BLIP-2

https://github.com/ymcui/Chinese-LLaMA-Alpaca

Open-source models
https://github.com/nichtdax/awesome-totally-open-chatgpt

ChatGLM-6B (Tsinghua), https://github.com/THUDM/ChatGLM-6B/blob/main/README_en.md

ChatYuan (ClueAI/元语智能), https://github.com/clue-ai/ChatYuan

RWKV ecosystem Hacker News, Twitter, Reddit, LoRA (weights: 7B, 14B), TextSynth Server CPU inference, Gradio demo, blog post, SpikeGPT

OpenAI
ChatGPT release notes

ChatGPT API (90% cost reduction) https://openai.com/blog/introducing-chatgpt-and-whisper-apis

GPT-4, https://openai.com/research/gpt-4, https://openai.com/product/gpt-4, https://arxiv.org/abs/2303.08774
GPTs are GPTs: An early look at the labor market impact potential of large language models, https://openai.com/research/gpts-are-gpts
Sparks of Artificial General Intelligence: Early experiments with GPT-4, https://arxiv.org/abs/2303.12712

OpenAI Evals, https://github.com/openai/evals

GitHub Copilot X, https://github.blog/2023-03-22-github-copilot-x-the-ai-powered-developer-experience/

Codex API discontinued, replaced by ChatGPT (turbo) API https://news.ycombinator.com/item?id=35242069, https://github.com/reasoning-machines/pal/pull/14/files

Other corporations
Microsoft: Bing Chat (search + browsing tool use, GPT-4 based, sort of ChatGPT plugin preview), Bing Image Creator (DALL-E based)

Anthropic: Claude

Google: PaLM API, Bard

Baidu: ERNIE Bot (文心一言) (Chinese language)

0 replies

alreadydone
on Mar 26
Maintainer
Author
Customization
https://twitter.com/johnjnay/status/1637843926840164353

-Supervised fine-tuning on your tasks
-Self-supervised learning (SSL) on your text

-RL w/ your reward model (RM)
-Filter high-temp outputs w/ RM
-Conditional SSL on RM-scored text

-Prompt w/ context
-Give it access to your tools
-Train (soft) parts of prompts

RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation (Microsoft / Wuhan), https://arxiv.org/abs/2303.12570

OpenChatKit, https://www.together.xyz/blog/openchatkit, featuring Customization recipes to fine-tune the model and Extensible retrieval system for live-updating answers.

ChatGPT plugins, https://openai.com/blog/chatgpt-plugins
Browsing, Code interpreter, Retrieval
Third party: Wolfram, etc.
https://www.toolkit.club/
https://github.com/team-openpm/openpm

Copilot for Docs, https://githubnext.com/projects/copilot-for-docs Compare https://github.com/context-labs/autodoc
Copilot for Pull Requests, https://githubnext.com/projects/copilot-for-pull-requests
Copilot for Your Codebase (WIP), https://githubnext.com/projects/copilot-view/
Copilot Chat, Voice, and Copilot for CLI

Tool use
Toolformer implementations: https://github.com/conceptofmind/toolformer (official), https://github.com/lucidrains/toolformer-pytorch

Tool Learning with Foundation Models (Tsinghua), https://arxiv.org/abs/2304.08354, https://github.com/OpenBMB/BMTools (Auto-GPT and BabyAGI support)

Augmented Language Models: a Survey, LeCun et al., https://arxiv.org/abs/2302.07842
Prompt Engineering: Augmented Language Models, Lilian Weng
mentions TALM, https://arxiv.org/abs/2205.12255

TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs (Microsoft), https://arxiv.org/abs/2303.16434
Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models (Microsoft), https://arxiv.org/abs/2303.04671, https://github.com/microsoft/TaskMatrix
HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace (Microsoft), https://arxiv.org/abs/2303.17580, https://github.com/microsoft/JARVIS

Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP (Stanford), https://github.com/stanfordnlp/dsp

Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback (Microsoft/Columbia U), https://arxiv.org/abs/2302.12813

ART: Automatic multi-step reasoning and tool-use for large language models (UWash, Microsoft, UCI, Allen, Meta), https://arxiv.org/abs/2303.09014

The surprising ease and effectiveness of AI in a loop, Matt Webb, https://interconnected.org/home/2023/03/16/singularity

A simple Python implementation of the ReAct pattern for LLMs, Simon Willison, https://til.simonwillison.net/llms/python-react-pattern
See also MM-ReAct: Prompting ChatGPT for Multimodal Reasoning and Action, https://multimodal-react.github.io/

Tool building (Coding)
Planning with Large Language Models for Code Generation, Tenenbaum and MIT-IBM Waston, https://arxiv.org/abs/2303.05510
Tenenbaum previously co-authored DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning, https://arxiv.org/abs/2006.08381
See also Self-planning Code Generation with Large Language Model (PKU), https://arxiv.org/abs/2303.06689

ViperGPT: Visual Inference via Python Execution for Reasoning (Columbia U), https://viper.cs.columbia.edu/

Reflexion: an autonomous agent with dynamic memory and self-reflection (MIT / Northeastern), https://arxiv.org/abs/2303.11366, blog post

Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks (Waterloo/UCSB/Google, Nov 2022), https://github.com/wenhuchen/Program-of-Thoughts
compare Program-aided Language Models, https://arxiv.org/abs/2211.10435

Symbolic Knowledge Distillation: from General Language Models to Commonsense Models, Yejin Choi et al., https://arxiv.org/abs/2110.07178

AI efficiency
In AI, is bigger always better?, Anil Ananthaswamy, https://www.nature.com/articles/d41586-023-00641-w

CoLT5: Faster Long-Range Transformers with Conditional Computation (Google), https://arxiv.org/abs/2303.09752

Algorithm optimization
EvoPrompting: Language Models for Code-Level Neural Architecture Search (NYU/Google Brain), https://arxiv.org/abs/2302.14838

Symbolic Discovery of Optimization Algorithms, Quoc Le et al. (Google/UCLA), https://github.com/lucidrains/lion-pytorch, https://arxiv.org/abs/2302.06675

Theorem proving
Baldur: Whole-Proof Generation and Repair with Large Language Models, First, Rabe, Ringer and Brun, https://arxiv.org/abs/2303.04910

Magnushammer: A Transformer-based Approach to Premise Selection, Albert Jiang, Szegedy, Yuhuai Wu et al., https://arxiv.org/abs/2303.04488

ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics (new dataset), Zhangir Azerbayev, Ayers, Avigad et al. https://github.com/zhangir-azerbayev/ProofNet

Future of Mathematics
Some thoughts on automation and mathematical research, Akshay Venkatesh, Nov 2021, https://www.math.ias.edu/~akshay/research/IASEssay.pdf
Fields Medal Symposium, Oct 2022 (pre-ChatGPT), http://www.fields.utoronto.ca/activities/22-23/fieldsmedalsym
Participants' impressions collected by Michael Harris https://siliconreckoner.substack.com/p/notes-from-the-2022-fields-medal

Terence Tao's integration of ChatGPT into daily workflow, picked up by Chinese news outlets
See also Would it be possible to create a tool to automatically diagram papers?

More
Learning to Compress Prompts with Gist Tokens, Jesse Mu et al., https://arxiv.org/abs/2304.08467

GPT-4: The Bitterer Lesson, Alberto Romero, https://thealgorithmicbridge.substack.com/p/gpt-4-the-bitterer-lesson

A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT (Vanderbilt), https://arxiv.org/abs/2302.11382 and ChatGPT Prompt Patterns for Improving Code Quality, Refactoring, Requirements Elicitation, and Software Design (Vanderbilt), https://arxiv.org/abs/2303.07839

Impact of Code Language Models on Automated Program Repair (Alberta / Purdue), https://arxiv.org/abs/2302.05020

Program synthesis in chip design, https://hub.baai.ac.cn/view/24406 (Chinese)

0 replies

distbit0
on Mar 27
I did reply to your twitter thread btw https://twitter.com/0xDist/status/1631883008918835201.

Maybe I am also shadow-banned 😅

I coincidentally both came across your twitter thread and this thread separately, but fully agree with both.

0 replies

alreadydone
on Mar 28
Maintainer
Author
Thanks for your reply @distbit0 ! I wrote the post on Feb 14 before you replied :) I like the neuroscience articles you linked to but I'm not sure what inspirations to get from them ... I had written some responses to your tweets but too much is going on these days; when I get around to polish them a bit I'll post them here.

0 replies

alreadydone
on Apr 11
Maintainer
Author
LLM agents
https://github.com/jtmuller5/The-HustleGPT-Challenge (inspred Auto-GPT and BabyAGI)

https://github.com/Torantulino/Auto-GPT (>100k stars, implements code execution/improvement)
local Auto-GPT: https://github.com/keldenl/gpt-llama.cpp/blob/master/docs/Auto-GPT-setup-guide.md#auto-gpt-setup-guide
https://www.aomni.com/ (research agent, can browse Internet!), tweet
https://www.cognosys.ai/create (button to toggle browsing)
https://github.com/reworkd/AgentGPT (web demo: https://agentgpt.reworkd.ai/)
https://github.com/yoheinakajima/babyagi babyagi.org
https://github.com/seanpixel/Teenage-AGI
Generative Agents: Interactive Simulacra of Human Behavior, Percy Liang et al., https://arxiv.org/abs/2304.03442
Experimenting with LLMs to Research, Reflect, and Plan, Eugene Yan, https://eugeneyan.com/writing/llm-experiments/
https://github.com/corca-ai/EVAL
https://github.com/pHaeusler/micro-agent
https://github.com/refcell/run-wild
Language Model Cascades, https://arxiv.org/abs/2207.10342

Foundation Models for Decision Making: Problems, Methods, and Opportunities (Google/Berkeley/MIT), https://arxiv.org/abs/2303.04129

https://github.com/hwchase17/langchain (recently got $10M investment): Agents

https://github.com/eumemic/ai-legion
https://github.com/mbusigin/yaml-runner

Lean 4 agent sagredo (code), receives feedback from a proof assistant, but no web browsing etc. yet. See also https://github.com/leanprover-community/repl.

Minecraft agents: Voyager (creates tools), GITM

Reinforcement learning
Reward Design with Language Models (Stanford/DeepMind), uses GPT3 API; proxy reward function is more akin to Constitutional AI than RLHF, https://arxiv.org/abs/2303.00001, news

Vision-Language Models (Flamingo) as Success Detectors (DeepMind), https://arxiv.org/abs/2303.07280

Reinforcement Learning from Passive Data via Latent Intentions (Berkeley), https://arxiv.org/abs/2304.04782

Reinforcement Learning for Language Models, Yoav Goldberg, https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81

Active learning
Internet Explorer: Targeted Representation Learning on the Open Web, Deepak Pathak et al. (CMU/Berkeley), https://internet-explorer-ssl.github.io/
Compare Socially situated artificial intelligence enables learning from human interaction, Li Fei-Fei et al., https://www.pnas.org/doi/10.1073/pnas.2115730119
See also Rewarding Chatbots for Real-World Engagement with Millions of Users (Chai Research), https://arxiv.org/abs/2303.06135

Active Self-Supervised Learning: A Few Low-Cost Relationships Are All You Need, LeCun et al., https://arxiv.org/abs/2303.15256

Iterated improvement
Examples of AI Improving AI, Thomas Woodside, https://ai-improving-ai.safe.ai/

AI for Science
AI for Science: An Emerging Agenda (Cambridge / Madison / Tübingen), https://arxiv.org/abs/2303.04217

Emergent autonomous scientific research capabilities of large language models, https://arxiv.org/abs/2304.05332, tweet (John Nay)

ChemCrow: Augmenting large-language models with chemistry tools, https://arxiv.org/abs/2304.05376, tweet (Jim Fan)


##Datasets for fine-tuning
https://github.com/yaodongC/awesome-instruction-dataset
https://github.com/PhoebusSi/Alpaca-CoT
https://github.com/JeremyAlain/imitation_learning_from_language_feedback#download-all-finetuning-datasets (language feedback)

##Toolformer / plugin / APIs:
https://github.com/ShishirPatil/gorilla
https://github.com/StevenGrove/GPT4Tools
https://github.com/danielgross/LlamaAcademy
https://github.com/teknium1/GPTeacher
https://huggingface.co/chavinlo/toolpaca
https://huggingface.co/kaiokendev/SuperCOT-LoRA

StackLLaMA: A hands-on guide to train LLaMA with RLHF, https://huggingface.co/blog/stackllama

Context length
Scaling Transformer to 1M tokens and beyond with RMT, https://arxiv.org/abs/2304.11062 (2M tokens)
https://hazyresearch.stanford.edu/blog/2023-03-27-long-learning
https://hazyresearch.stanford.edu/blog/2023-03-07-hyena
MPT-7B-StoryWriter-65k+
100k context window, Claude (Anthropic)
https://github.com/BlinkDL/RWKV-LM/ (RNN, attention-free infinite context)
