import os
import time

from rich.console import Console
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from flashrank import Ranker, RerankRequest



console = Console()

# modelo que vai responder as perguntas
MODELO_NOME = "llama3.2"

llm = ChatOllama(model=MODELO_NOME)

# usamos o msm modelo pra embeddings so pra simplificar
embeddings = OllamaEmbeddings(model=MODELO_NOME)

# modelo de reranking (nao gera texto)
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")


# carrega os pdfs e monta o banco vetorial

def carregar(pasta_arquivos):

    # pega so os pdfs da pasta
    arquivos_pdf = [
        f for f in os.listdir(pasta_arquivos)
        if f.lower().endswith(".pdf")
    ]

    if not arquivos_pdf:
        return [], None

    console.print(f"[cyan]Carregando {len(arquivos_pdf)} PDFs...[/cyan]")

    documentos = []

    # leitura pagina por pagina
    for arquivo in arquivos_pdf:
        caminho = os.path.join(pasta_arquivos, arquivo)
        loader = PyPDFLoader(caminho)

        # cada pagina vira um Document
        documentos.extend(loader.load())

    # pdf inteiro nao cabe no contexto do modelo
    # entao quebramos em pedaços menores
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # tamanho aproximado
        chunk_overlap=200   # sobreposicao pra nao perder sentido
    )

    documentos_divididos = splitter.split_documents(documentos)

    # cria o banco vetorial
    # cada chunk vira um embedding
    vectorstore = Chroma.from_documents(
        documents=documentos_divididos,
        embedding=embeddings
    )

    # retriever faz a busca semantica
    # k=20 pq depois vamos reranquear
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )

    console.print("[green]Base vetorial pronta.[/green]")
    return documentos, retriever


# ------ modos de chat ---------

# modo mais simples: pergunta vai direto pro modelo
def chat_simples(pergunta):

    inicio = time.time()
    resposta = llm.invoke(pergunta)
    tempo = time.time() - inicio

    console.print("\n[bold blue]Resposta:[/bold blue]")
    console.print(resposta.content)
    console.print(f"[dim]Tempo: {tempo:.2f}s[/dim]\n")


# modo força bruta: manda tudo dos pdfs
def chat_full(pergunta, documentos):

    # junta o texto de todas as paginas
    contexto = "\n\n".join(doc.page_content for doc in documentos)

    # monta prompt com contexto
    prompt = ChatPromptTemplate.from_template(
    "Use apenas as informações do contexto abaixo para responder.\n"
    "Se a resposta não estiver no contexto, diga que não encontrou.\n\n"
    "Contexto:\n{contexto}\n\n"
    "Pergunta: {pergunta}"
)


    prompt_formatado = prompt.format(
        contexto=contexto,
        pergunta=pergunta
    )

    inicio = time.time()
    resposta = llm.invoke(prompt_formatado)
    tempo = time.time() - inicio

    console.print("\n[bold blue]Resposta (Full Context):[/bold blue]")
    console.print(resposta.content)
    console.print(f"[dim]Tempo: {tempo:.2f}s[/dim]")
    console.print(f"[dim]Tamanho do contexto: {len(contexto)} chars[/dim]\n")


# modo RAG + reranking
def chat_rag(pergunta, retriever):

    inicio = time.time()

    # busca semantica inicial
    docs_recuperados = retriever.invoke(pergunta)

    # FlashRank espera id + texto
    passages = []
    for i, doc in enumerate(docs_recuperados):
        passages.append({
            "id": i,
            "text": doc.page_content
        })

    # pedido de reranking
    rerank_request = RerankRequest(
        query=pergunta,
        passages=passages
    )

    resultados = ranker.rerank(rerank_request)

    # pegamos so os 5 melhores
    melhores_trechos = []
    for item in resultados[:5]:
        melhores_trechos.append(passages[item["id"]]["text"])

    contexto = "\n\n".join(melhores_trechos)

    # monta prompt com contexto
    prompt = ChatPromptTemplate.from_template(
    " Você deve responder APENAS com base no contexto fornecido."
    "Use apenas as informações do contexto abaixo para responder.\n"
    "Se a resposta não estiver no contexto, diga explicitamente:"
    "A informação não está presente nos documentos fornecidos."
    "Se a resposta não estiver no contexto, diga que não encontrou.\n\n"
    "Contexto:\n{contexto}\n\n"
    "Pergunta: {pergunta}"
)


    prompt_formatado = prompt.format(
        contexto=contexto,
        pergunta=pergunta
    )

    resposta = llm.invoke(prompt_formatado)
    tempo = time.time() - inicio

    console.print("\n[bold blue]Resposta (RAG):[/bold blue]")
    console.print(resposta.content)
    console.print(f"[dim]Tempo: {tempo:.2f}s[/dim]")
    console.print(f"[dim]Chunks usados: {len(melhores_trechos)}[/dim]")
    console.print(f"[dim]Contexto final: {len(contexto)} chars[/dim]\n")


# -------- programa principal ---------

if __name__ == "__main__":

    pasta_doc = "documentos"

    # carrega tudo so uma vez
    documentos, retriever = carregar(pasta_doc)

    while True:
        console.print("\n[bold yellow]--- MENU ---[/bold yellow]")
        console.print("1 - Chat simples")
        console.print("2 - Chat com contexto completo")
        console.print("3 - Chat RAG")
        console.print("4 - Sair")

        opcao = console.input("\nEscolha uma opcao: ")

        match opcao:
            case "1":
                pergunta = console.input("\nDigite sua pergunta: ")
                chat_simples(pergunta)

            case "2":
                if documentos:
                    pergunta = console.input("\nDigite sua pergunta: ")
                    chat_full(pergunta, documentos)
                else:
                    console.print("[red]Nenhum PDF carregado.[/red]")

            case "3":
                if retriever:
                    pergunta = console.input("\nDigite sua pergunta: ")
                    chat_rag(pergunta, retriever)
                else:
                    console.print("[red]Nenhum PDF carregado.[/red]")

            case "4":
                console.print("[green]Encerrando.[/green]")
                break

            case _:
                continue
