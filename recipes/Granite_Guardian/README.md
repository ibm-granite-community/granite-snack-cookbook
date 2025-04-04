# README for Guardian

This directory contains various granite guardian cookbooks in generative AI for detection of risks in both user prompt and AI assistant response.

Here's the summary of the cookbooks:

* [Hate, Abuse, and Profanity (HAP) Detection](HAP.ipynb) : 38M model for detecting hate, abuse, and profanity.

* Granite Guardian 3.0. NOTE: These recipes currently require a GPU. If you are trying them on Google Colab, you must select a 'T4 GPU' for your runtime:

  * [Quick Start Guide](Granite_Guardian_Quick_Start.ipynb): Explore how 2B/8B models are used to identify potential risks in prompts and responses across dimensions such as social bias, profanity, violence, sexual content, unethical behavior, jailbreaking, and groundedness/relevance for Retrieval-Augmented Generation.
  * [Detailed Guide](Granite_Guardian_Detailed_Guide.ipynb): Explore in-depth various use cases that highlight different risk in prompts and responses across various risk dimensions of our taxonomy. This walkthrough also demonstrates how to integrate your custom risk definitions into Granite Guardian.
  * [Usage Governance Workflow](Granite_Guardian_Usage_Governance_Workflow.ipynb): Explore the use of Granite Guardian together with IBM's AI Risk Atlas to evaluate the risk profile of a potential AI use case.  This walkthrough shows how to use risk definitions from the IBM AI Risk Atlas to augment the Granite Guardian results.
