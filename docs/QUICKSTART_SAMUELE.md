# Quickstart per Samuele — v0.1.0

La libreria si collega **dopo** la tua joint discovery/reconstruction. Legge il joint e q(t),
poi le forze osservate nello stesso frame metrico e nello stesso clock.
Non richiede di rilanciare Gaussian Splatting. La validazione fisica sui tuoi dati reali resta da fare.

## 1. Installa e prova il collegamento sintetico

Dopo la creazione della repository privata e la concessione dell'accesso:

```bash
git clone git@github.com:andrearichichi/ForceArtGS-Physics.git
cd ForceArtGS-Physics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python examples/03_minimal_numpy.py
```

È una demo sintetica CPU, senza dati esterni. Per verificare il contratto con un output ForceArtGS
sintetico già incluso, senza training:

```bash
python examples/01_forceartgs_end_to_end.py --methods IntegralLS --out outputs/synthetic_check
```

Il parquet viene usato se pyarrow è installato; altrimenti questa **sola demo sintetica** usa gli stessi array sintetici.
Per leggere un vero parquet dal tuo codice: `python -m pip install '.[real]'`.

## 2. Collega un episodio misurato

Servono due input della **stessa interazione**:

1. La tua directory `joint_authority/v000` con `authority.json` e `joint_states.npz` (Level A).
   In alternativa una directory di discovery con `joint_candidates.json` (Level B).
2. Un JSON con le forze osservate già calibrate, sincronizzate e trasformate nel frame del joint.
   Producilo dal tuo preprocessing del robot; la libreria non inventa TCP, bias, segno o lag.

Imposta i tre percorsi e il nome del frame **verificati per il tuo episodio**:

```bash
export FA_RUN=/percorso/assoluto/al/branch/joint_authority/v000
export FA_FORCES=/percorso/assoluto/alle/forze_osservate.json
export FA_FRAME=nome_del_frame_metrico_verificato

forceartgs-physics diagnose \
  --forceartgs-run "$FA_RUN" --level A \
  --forces-json "$FA_FORCES" --frame "$FA_FRAME"

python examples/04_samuele_pipeline.py \
  --run "$FA_RUN" --level A \
  --forces-json "$FA_FORCES" --frame "$FA_FRAME" \
  --method AKF --out outputs/measured_episode_diagnostic.json
```

Se usi la discovery, cambia il percorso e `--level B` in entrambi i comandi.
Il secondo comando si ferma se il diagnostico non identifica separatamente I e c.
Non usa forze sintetiche e non sceglie calibrazioni di default. L'output è un **rapporto diagnostico**:
non certifica che il modello viscoso descriva il tuo oggetto reale.

Formato richiesto e snippet per scrivere il JSON dai tuoi array:
[contratto delle forze osservate](DATA_FORMATS.md#json-delle-forze-osservate-per-gli-episodi-misurati).
La CLI richiede provenienza di frame, segno, assi, contatto/TCP, tare e sincronizzazione.
Se questi dati mancano, completa il preprocessing del tuo episodio prima di identificare la fisica.

## 3. Export fisico dopo la revisione del modello

Solo dopo aver controllato calibrazione, sincronizzazione, adeguatezza del modello viscoso e predizione
su un'altra interazione misurata, puoi usare questo codice nel tuo ambiente:

```python
from forceartgs_physics import api
from forceartgs_physics.schemas import IdentificationResult
from forceartgs_physics.export import write_dynamics_yaml

result = IdentificationResult.load("outputs/measured_episode_diagnostic.json")
api.save_parameters(result, "outputs/physical_parameters.json")
write_dynamics_yaml(result, "outputs/asset/physics/dynamics.yaml")
```

Entrambi gli export rifiutano risultati falliti, non identificabili, non diagnosticati o con parametri non finiti/positivi.
La revisione dell'adeguatezza del modello resta responsabilità dell'esperimento: il CRLB da solo non la dimostra.
Massa del link, CoM e tensore d'inerzia restano `null`. La massa efficace traslazionale o l'inerzia sull'asse
sono nel blocco `joint_space_identification`, insieme al damping e alla provenienza.

`mode: identified` riguarda i parametri del joint; il file non certifica un asset rigido completo.
Un simulatore che richiede massa e tensore del link necessita di misure indipendenti per i campi `null`.
