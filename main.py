import os

from rich.console import Console
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from flashrank import Ranker, RerankRequest


console = Console()

# config do modelo
MODELO_NOME = "llama3.2"

# modelo de linguagem 
llm = ChatOllama(model=MODELO_NOME)

# Transformar texto em vetores com embeddings 
# Usamos o mesmo modelo por simplicidade
embeddings = OllamaEmbeddings(model=MODELO_NOME)

# Modelo de reranking. Ele não gera texto, só reordena trechos com base na pergunta
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")


# carregando agora para nao travar o menu
# lê todos os pdfs, quebra o texto em pedaços, cria o banco vetorial retorna documentos completos (para o modo Full Context) e retriever pronto para o RAG
def carregar(pasta_arquivos):

    # lista os arquivos PDF da pasta
    arquivos_pdf = [
        f for f in os.listdir(pasta_arquivos)
        if f.lower().endswith(".pdf")
    ]

    if not arquivos_pdf:
        return [], None

    console.print(f"[cyan]Carregando {len(arquivos_pdf)} arquivos PDF...[/cyan]")

    documentos = []

    # leitura
    for arquivo in arquivos_pdf:
        caminho = os.path.join(pasta_arquivos, arquivo)
        loader = PyPDFLoader(caminho)

        # cada PDF vira uma lista de páginas (document)
        documentos.extend(loader.load())

    # PDFs grandes não cabem inteiros no contexto do modelo
    # então quebramos em pedaços menores
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # tamanho de cada pedaço
        chunk_overlap=200   # sobreposição para não perder contexto
    )

    documentos_divididos = splitter.split_documents(documentos)

    # banco vetorial
    # cada pedaço vira um embedding e é armazenado
    vectorstore = Chroma.from_documents(
        documents=documentos_divididos,
        embedding=embeddings
    )

    # Retriever faz a busca semântica
    # k=20: buscamos mais trechos do que o necessario, pois depois eles serão reranqueados
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )

    console.print("[green]Base de dados preparada com sucesso.[/green]")
    return documentos, retriever


# ---- modos ----

# nao usa pdf, a pergunta vai direto pro modelo
def chat_simples(pergunta):
    resposta = llm.invoke(pergunta)
    console.print(f"\n[bold blue]Resposta:[/bold blue]\n{resposta.content}\n")


# usa todo o conteudo dos pdfs como contexto e envia tudo pro modelo
def chat_full(pergunta, documentos):

    # junta o conteudo de todas as paginas
    contexto = "\n\n".join(doc.page_content for doc in documentos)

    # cria o template do prompt
    prompt = ChatPromptTemplate.from_template(
        "Contexto:\n{contexto}\n\nPergunta: {pergunta}"
    )

    # preenche o template com os dados reais
    prompt_formatado = prompt.format(
        contexto=contexto,
        pergunta=pergunta
    )

    # envia o texto final diretamente para o modelo
    resposta = llm.invoke(prompt_formatado)

    console.print(
        f"\n[bold blue]Resposta (Full Context):[/bold blue]\n{resposta.content}\n"
    )


# usa RAG + FlashRank para selecionar os melhores trechos
def chat_rag(pergunta, retriever):

    # busca semantica inicial
    docs_recuperados = retriever.invoke(pergunta)

    # FlashRank espera uma lista de passagens com id e texto
    passages = []
    for i, doc in enumerate(docs_recuperados):
        passages.append({
            "id": i,
            "text": doc.page_content
        })

    # monta a requisição de reranking
    rerank_request = RerankRequest(
        query=pergunta,
        passages=passages
    )

    # executa o reranking
    resultados = ranker.rerank(rerank_request)

    # seleciona apenas os 5 melhores trechos após o reranking
    melhores_trechos = []
    for item in resultados[:5]:
        melhores_trechos.append(passages[item["id"]]["text"])

    # contexto final enviado ao modelo
    contexto = "\n\n".join(melhores_trechos)

    prompt = ChatPromptTemplate.from_template(
        "Contexto recuperado:\n{contexto}\n\nPergunta: {pergunta}"
    )

    prompt_formatado = prompt.format(
        contexto=contexto,
        pergunta=pergunta
    )

    resposta = llm.invoke(prompt_formatado)

    console.print(
        f"\n[bold blue]Resposta (RAG + FlashRank):[/bold blue]\n{resposta.content}\n"
    )


# ---- MAIN ----

if __name__ == "__main__":

    pasta_doc = "documentos"

    # preparação inicial 
    documentos, retriever = carregar(pasta_doc)

    while True:
        console.print("\n[bold yellow]--- MENU ---[/bold yellow]")
        console.print("1 - Chat simples")
        console.print("2 - Chat com contexto completo (PDF inteiro)")
        console.print("3 - Chat RAG")
        console.print("4 - Sair")

        opcao = console.input("\nEscolha uma opção: ")

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
                console.print("[green]Encerrando o programa.[/green]")
                break

            case _:
                continue
