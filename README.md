# Chat com LLM + PDFs (RAG)

Esse projeto é um chat em Python usando LLM local (Ollama), com suporte a perguntas
baseadas em PDFs usando RAG.

>  **Aviso**  
> Este README **NÃO é o relatório final do trabalho**.  
> Ele é apenas um resumo do que foi feito até agora, pensado para facilitar a comunicação entre os membros da equipe.  
> Quando o trabalho estiver finalizado, esse documento deve mudar .

---

## 🛠 Tecnologias utilizadas

- ![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) **Python 3.10+**
- ![LangChain](https://img.shields.io/badge/LangChain-Framework-green?logo=chainlink) **LangChain**
- ![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black) **Ollama (Llama 3.2)**
- ![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple) **ChromaDB**
- **FlashRank** (reranking no RAG)

---

##  Como executar 

1. Ter **Python 3.10+** instalado  
2. Instalar o **Ollama**
3. Baixar o modelo:
   ```bash
   ollama pull llama3.2
    ```
4. Instale as dependencias:
    ```bash
   pip install -r requirements.txt
    ```
5. Colocar os pdf na pasta de documentos (  usei o pdf da aula como teste [ia2.pdf](documentos/ia2.pdf) )

## 📁 Estrutura do projeto

- `main.py`  
  Arquivo principal do projeto.  
  Aqui fica:
  - o menu de interação
  - a lógica dos três tipos de chat
  - a parte de RAG
  - a integração com o modelo local

- `documentos/`  
  Pasta onde devem ser colocados os PDFs.  
  Tudo que estiver aqui é carregado automaticamente quando o programa inicia.

- `logs_interacao_ia.txt`  
  Arquivo com o registro das interações com IA durante o desenvolvimento.  
  Funciona como um diário técnico, documentando decisões, dúvidas e ajustes feitos,
  conforme solicitado nas especificações do trabalho .

- `requisitos.txt`  
  Lista das bibliotecas necessárias para executar o projeto.

---

##  Como o código foi pensado (resumo geral)

O trabalho exigia **três tipos de chat**, então o código foi estruturado para atender
cada um deles separadamente:

### 1.  Chat simples
- Apenas envia a pergunta diretamente para o modelo
- Não usa PDFs
- Serve como base de comparação

### 2. Chat com contexto completo (Full Context)
- Junta todo o texto dos PDFs em um único contexto
- Envia tudo de uma vez para o modelo

### 3. Chat RAG
- Os PDFs são quebrados em pequenos trechos
- Cada trecho vira um embedding
- Um banco vetorial armazena esses embeddings
- Na pergunta, apenas os trechos mais relevantes são usados

Alem disso:
- O banco vetorial é criado uma única vez no início do programa, evitando lentidão a cada nova pergunta.
- O RAG recupera mais trechos do que o necessário e refina essa seleção depois

---


## Referências

- NETWORK CHUCK. *Build a RAG system with LangChain (from scratch)*. YouTube, 2023.  
  Disponível em: https://www.youtube.com/watch?v=E4l91XKQSgw&t=1096s  
  Acesso em: 3 jan. 2026.

- LANGCHAIN. *LangChain Documentation*.  
  Disponível em: https://python.langchain.com/  
  Acesso em: 3 jan. 2026.

- LANGCHAIN. *Ollama integration*.  
  Disponível em: https://python.langchain.com/docs/integrations/llms/ollama/  
  Acesso em: 3 jan. 2026.

- OLLAMA. *Ollama: run large language models locally*.  
  Disponível em: https://ollama.com/  
  Acesso em: 3 jan. 2026.

- LANGCHAIN. *Text embeddings*.  
  Disponível em: https://python.langchain.com/docs/concepts/text_embeddings/  
  Acesso em: 3 jan. 2026.

- CHROMA. *Chroma Documentation*.  
  Disponível em: https://docs.trychroma.com/  
  Acesso em: 3 jan. 2026.

- FLASHRANK. *FlashRank: fast reranking for retrieval-augmented generation*.  
  Disponível em: https://github.com/PrithivirajDamodaran/FlashRank  
  Acesso em: 3 jan. 2026.
