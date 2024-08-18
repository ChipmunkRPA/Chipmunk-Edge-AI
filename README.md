I used not to believe in open-source GenAI LLM models because I don’t believe they can ever catch up to the closed-source models OpenAI and Anthropic own. I was wrong. 

Three companies blow up my mind with their open-source GenAI LLM model performance and offerings. Namely, the almighty Google with its Gemini collection, the unstoppable Meta with its llama 3 series, and Europe’s hope in the race of AI - Mistral. A few weeks ago (on the last day of July 2024), Google open-sourced and launched its smallest model within the Gemini family - Gemma 2 2B. Not only does the model run on almost virtually any device, given its extremely lightweight 2-billion parameters, but it outperforms ChatGPT 3.5 on multiple fronts!

It got me thinking: why don’t I create a local GenAI LLM solution using Gemma 2 2B that can run purely 100% locally WITHOUT a need to pull any third-party API online? Even better, instead of a simple local AI chatbot, why don’t I create a 100% local RAG AI system so that users can process their own local documents to give the local AI knowledge of the private documents (similar to the way I built Confide A.I.)?

To make the dream come true, I know I need to set up a local server within the user’s local device (100% self-contained). I decided to use Ollama as my dependency to achieve this goal. Ollama is an open-source server that hosts large language models (LLMs) locally on user’s machines.

Now, within two weeks of heavy engineering, I have it built and decided to name it Chipmunk Edge AI, representing the combination of edge computing (100% local on the endpoint devices such as your consumer-grade laptops and even your cellphones without a need for internet to use the AI functionalities) and the GenAI LLM abilities.

Because all the LLM operations are 100% local, it means:

1. Your documents and chats will never be exposed to a third-party AI provider when using the application. The level of confidentiality and privacy of Chipmunk Edge AI is unparalleled.

2. Hence, this application is perfect for companies/government agencies with the toughest confidentiality and privacy requirements (e.g., companies/agencies have to comply with HIPPA and FedRAMP etc.).

3. No AI usage costs whatsoever. Because you run the AI model completely within your own machine, no one charges you additional fees!

4. Due to the lightweight yet excellent performance of the Gemma 2 2B, as long as your device has more than 8 GB RAM, you can totally run the entire application on your local device! No GPU is even required.

Do you want to give it a try? Download it on my website [chipmunkrpa.com](https://www.chipmunkrpa.com/a-i) at absolutely ZERO cost.

Want to hear some better news? Not only did I make the application free to everyone, it was also open-sourced to everyone. Check out this repo!

My Next Big Thing:

My team is going to release a professional AI writing platform (Inkwise) to help professionals tackle the most difficult writing tasks in the world (e.g., investor relation scripts, technical accounting and tax memos, legal memos and briefs, deep marketing research and investigative journals). Sign up our waitlist here https://www.inkwise.ai/
