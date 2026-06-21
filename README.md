# AI+

**Assistente AI ibrido** — routing intelligente tra modelli locali (Ollama) e cloud (OpenAI, OpenRouter, Gemini, opencode).

Agenti con tool (leggi/scrivi/modifica file, esegui comandi, cerca sul web), RAG, note in stile Obsidian, e interfaccia web completa.

## Requisiti

- **Python 3.10 o superiore**
- Opzionale: [Ollama](https://ollama.com) per modelli AI locali
- Opzionale: chiavi API per provider cloud (OpenAI, OpenRouter, Gemini)

## Installazione

### Metodo rapido — `setupAI+`

```bash
python3 setupAI+
```

Lo script guida passo passo:
1. Verifica/installa Python 3.10+
2. Crea ambiente virtuale
3. Scarica il codice sorgente
4. Installa tutte le dipendenze
5. Crea i comandi `hy` e `ai-plus`
6. Installa Ollama (opzionale)
7. Configurazione iniziale automatica

Funziona su **Windows**, **macOS**, **Linux** e **Raspberry Pi**.

### Genera il pacchetto di installazione

Dalla directory del progetto:

```bash
hy pack
```

Produce `ai-plus-install-pack.zip` contenente: setupAI+, setup.py, requirements.txt, README.md, e l'intero codice sorgente. Copia lo zip su qualsiasi macchina ed esegui:

```bash
unzip ai-plus-install-pack.zip
python3 setupAI+
```

### Da sorgente (sviluppo)

```bash
git clone https://github.com/ai-plus/ai-plus.git
cd ai-plus

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

## Comandi

| Comando | Descrizione |
|---------|-------------|
| `hy 'prompt'` | Chat da terminale |
| `hy web` | Avvia interfaccia web su http://localhost:8081 |
| `hy pack` | Genera il pacchetto di installazione (.zip) |
| `hy generate 'prompt'` | Output JSON strutturato |
| `hy learn from-dir <path>` | Indicizza documenti nella knowledge base |
| `hy model list` | Elenca modelli disponibili |
| `hy config show` | Mostra configurazione corrente |
| `hy notes list` | Gestione note in stile Obsidian |
| `hy project create <nome>` | Scaffold progetto da template |

## Modello locale (Ollama)

```bash
ollama pull llama3         # o altro modello
ollama serve               # avvia il server
hy 'Ciao!'                 # usa il modello locale
```

## Web UI

```bash
hy web --port 5000
```

Apri `http://localhost:8081`. La UI include:
- Chat con routing automatico locale/cloud
- Agenti con tool calling integrati
- Knowledge base RAG
- Note in stile Obsidian
- Dashboard statistiche
- Terminale integrato
- Progetti scaffolding

## Distribuire su altre macchine

```bash
# Sulla macchina di sviluppo
hy pack

# Copia lo zip sulla macchina di destinazione (PC, Mac, Linux, Raspberry Pi)
# Sulla macchina di destinazione
unzip ai-plus-install-pack.zip
python3 setupAI+
```

## Licenza

MIT
