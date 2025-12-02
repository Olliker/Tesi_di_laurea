# Tesi: Motore di previsione vendite per ErpAI

Repository della tesi sviluppata presso Spazio Dev S.r.l. Il lavoro descrive la progettazione e l’implementazione di un modulo di previsione vendite (Python + FastAPI) integrato nel gestionale \textit{ErpAI}, con funzionalità di explainability basate su SHAP.

## Obiettivi e contributi
- Pulizia e trasformazione dei dati storici di fatturazione in feature utili.
- Addestramento e valutazione di modelli di machine learning per stimare la domanda futura.
- Esposizione delle previsioni tramite API REST integrate con l’ERP.
- Spiegazione delle predizioni (XAI) per supportare decisioni trasparenti.
- Processo di sviluppo documentato con metodologie agili e pipeline di test.

## Struttura della tesi
- `chapters/1_introduction.tex`: contesto aziendale e idea progettuale.
- `chapters/2_processes.tex`: processi aziendali, strumenti e metodologie.
- `chapters/3_stage_desc.tex`: descrizione dello stage e analisi dei rischi.
- `chapters/4_requirement_analysis.tex`: requisiti funzionali e non funzionali.
- `chapters/5_design.tex`: architettura, pipeline dati e modello predittivo.
- `chapters/6_code.tex`: organizzazione del codice e componenti principali.
- `chapters/7_testing_ci.tex`: strategia di test e integrazione continua.
- `chapters/8_conclusions.tex`: risultati e possibili evoluzioni.

## Struttura del repository
- `thesis/files/chapters/`: testo della tesi.
- `thesis/files/code/`: snippet di codice inclusi nel documento.
- `thesis/files/img/`: figure e schemi.
- `thesis/files/preface/`: front matter (titolo, abstract, indice).
- `thesis/files/config/`: configurazioni LaTeX e variabili del documento.
- `thesis/files/thesis.tex`: entry point per la compilazione.

## Come compilare il PDF
Prerequisiti: distribuzione TeX completa (latexmk, pdflatex/lualatex), Pygments per gli highlight di `minted`.

1. Posizionarsi nella cartella dei sorgenti:
   ```bash
   cd thesis/files
   ```
2. Compilare con latexmk (shell-escape già configurato in `latexmkrc`):
   ```bash
   latexmk -pdf thesis.tex
   ```
3. Per pulire i file temporanei:
   ```bash
   latexmk -c
   ```

Il PDF generato si trova in `thesis/files/thesis.pdf`.
