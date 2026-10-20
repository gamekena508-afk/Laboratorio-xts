# Voice Lab — meu laboratório de voz com IA, de graça

Oi! Eu sou o Julian. Este é o laboratório de voz que eu fui construindo aos poucos, com muito teste e muito bug no caminho. Resolvi deixar ele aberto, **grátis**, para quem quiser testar, estudar ou melhorar.

É um servidor local com uma interface web: você escreve um texto, escolhe uma voz de referência, e o programa devolve o áudio. Roda no seu computador, no navegador, em `http://localhost:8000`.

> **Honestidade primeiro:** é um projeto em evolução. Tem limitações (veja mais abaixo) e eu não testei tudo em todos os sistemas. Se você achar um problema, abra uma *issue*. Isso ajuda de verdade.

## O que está incluído

A interface tem estas abas (os nomes são os que aparecem na tela):

| Aba | O que faz |
|---|---|
| **Síntese** | Gera voz a partir de texto e uma voz de referência. Você escolhe idioma e um preset: Controlado, Natural ou Expressivo. A prévia tem player com pausar e barra de tempo. |
| **Lab de Criação** | Cria vozes novas (nome, texto de prévia e idioma) a partir de vozes âncora. |
| **Extrator YT / Download YT** | Baixa áudio de um link do YouTube para usar como referência. |
| **Sampling Maker** | Gera samples a partir de um áudio de entrada, com efeito de remix e mistura com um segundo áudio (opcionais). |
| **Restaurar** | Restaura áudio com ruído: trecho (início e fim), força do denoise, correção de pitch e de tempo. Usa `ffmpeg`. |
| **Fusão** | Combina representações de vozes (não é recortar e colar dois áudios; o resultado precisa ser ouvido). |
| **Biometria** | Cadastra uma voz e compara outras amostras com ela por similaridade. |
| **Biblioteca** | Lista, ouve, envia e apaga as suas vozes. |
| **Configurações** | Depende de um serviço auxiliar que **não vem neste repositório** (veja Limitações). |

Arquivos do projeto:

- `server.py`: o servidor (FastAPI).
- `app.html`: a interface.
- `separador_engine.py`: separação de voz e música.
- `requirements.txt`: as dependências Python.
- `INICIAR-TTS.sh` / `start.bat`: iniciam o servidor (Linux / Windows).
- `PREPARAR-MODELO.sh` / `PREPARAR-MODELO.bat`: baixam o modelo de voz.
- `LICENSE` e `LICENCAS.md`: licenças.

## Antes de instalar

- **Python 3.10 ou 3.11.** O Python 3.12 ou mais novo não funciona com a biblioteca de voz usada.
- **ffmpeg** instalado no sistema e no `PATH`.
- **Placa NVIDIA** é o recomendado. O `requirements.txt` usa o PyTorch para CUDA 13. Se a sua CUDA for outra, troque a linha `--extra-index-url` pelo endereço certo em <https://pytorch.org/get-started/locally/>.
- **Espaço e internet:** o modelo tem cerca de 1,8 GB, fora o PyTorch. A internet é necessária no primeiro uso (modelo e detector de voz).
- **Git é opcional.** Dá para baixar o ZIP direto do GitHub.

### Como baixar o código

1. **Com Git, a primeira vez:** `git clone https://github.com/gamekena508-afk/Laboratorio-xts.git` baixa o projeto inteiro para uma pasta nova.
2. **Com Git, depois:** dentro da pasta, `git pull` traz só o que mudou. Use para atualizar uma cópia que você já tem.
3. **Sem Git:** na página do repositório no GitHub, clique em **Code** e depois em **Download ZIP**. Extraia numa pasta.

## Instalação rápida no Linux

```bash
cd Laboratorio-xts
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
./PREPARAR-MODELO.sh     # baixa o modelo; a biblioteca pergunta a licença e VOCÊ responde
./INICIAR-TTS.sh         # sobe em http://127.0.0.1:8000 e abre o navegador
```

## Instalação rápida no Windows

Abra o terminal (PowerShell) dentro da pasta do projeto:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
PREPARAR-MODELO.bat      # baixa o modelo; você responde a pergunta da licença
start.bat                # sobe em http://localhost:8000
```

Instale o `ffmpeg` e confira com `ffmpeg -version`. **Eu não testei a instalação limpa no Windows.** Se algo falhar, abra uma issue com a mensagem de erro.

## Como usar

1. Rode o iniciador. Quando o servidor subir, abra `http://localhost:8000`.
2. **Coloque a sua voz de referência:** um WAV com fala limpa de 10 a 30 segundos, de uma voz que você tem direito de usar. Copie para a pasta `reference_voices/` ou envie pela aba **Biblioteca**.
3. Na aba **Síntese**, escolha a voz, escreva o texto, selecione o idioma e o preset, e gere.
4. Comece com uma frase curta e ouça o resultado antes de produzir textos longos.
5. Nas outras abas, teste uma de cada vez. Cada uma depende de coisas diferentes (ffmpeg, internet, separação de áudio).

## Vozes

**Este repositório não traz nenhuma voz.** Eu só publico o que tenho certeza de que posso redistribuir. Use vozes suas, ou de pessoas que autorizaram. Clonar a voz de alguém sem permissão é problema seu, não do programa.

O **Lab de Criação** procura vozes âncora pelo nome dentro de `reference_voices/`. Se elas não existirem, o código usa vetores aleatórios **sem avisar**, e o resultado fica diferente do que eu mostro nos vídeos.

## Privacidade e segurança

- Os iniciadores sobem o servidor só na sua máquina (`127.0.0.1`). O `python server.py` direto também.
- O servidor **não tem senha** e aceita chamadas de qualquer origem (CORS `*`). Por isso **não exponha a porta 8000 na rede ou na internet**.
- Vozes fundidas e cadastros de biometria ficam em arquivos `.pkl`, lidos com `pickle`, que pode executar código. **Abra só arquivos que você mesmo gerou.**
- A interface carrega uma fonte do Google, e o detector de voz (Silero) é baixado pelo `torch.hub` na primeira chamada.
- O YouTube tem termos próprios. Use a extração só com conteúdo que você tem direito de usar.

## Limitações conhecidas

- A aba **Configurações** e o chat chamam um serviço auxiliar em `127.0.0.1:8011` que **não está neste repositório**. Sem ele, essas partes mostram erro.
- O `requirements.txt` vem de um ambiente que eu usei; **nunca foi instalado do zero a partir deste repositório**.
- `audio-separator` consta nas dependências, mas pode precisar de ajustes na sua máquina.
- Windows e macOS: não testados por mim.
- Os modelos de terceiros não estão aqui. Você baixa na fonte oficial.

## Comunidade

Achou um bug, teve uma ideia ou melhorou alguma coisa? Abra uma *issue* ou mande um *pull request*. Quanto mais claro o relato (o que você fez, o que esperava, o que aconteceu, a mensagem de erro), mais fácil de resolver.

## Licença

O **código deste repositório** é MIT. Veja [`LICENSE`](LICENSE).

O **modelo de voz** (XTTS v2) tem licença própria, a Coqui Public Model License, que **só permite uso não comercial**, inclusive para o áudio gerado. A MIT do código não muda isso. Os detalhes e as licenças das dependências estão em [`LICENCAS.md`](LICENCAS.md).

## Aviso

O programa é fornecido como está, sem garantia. Use com responsabilidade e só com vozes e conteúdos que você tem direito de usar.
