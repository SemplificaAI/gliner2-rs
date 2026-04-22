# GLiNER2-rs: Piano di Ottimizzazione IOBinding (Zero-Copy VRAM)

**Branch Attuale:** `Exp_IOBinding`
**Obiettivo:** Ridurre la latenza di inferenza eliminando i trasferimenti di memoria sul bus PCIe (CPU <-> GPU/NPU) tra i vari step della pipeline frammentata di GLiNER2, mantenendo i tensori residenti nella memoria del device (VRAM).

---

## ✅ Fase 1: Completata (Architettura Rust e Supervisor)
Abbiamo gettato le basi strutturali nel codice Rust per supportare l'esecuzione "Dual-Mode" (Modalità 2 IOBinding e Modalità 1 Standard di fallback).

1. **Gestione Controllata degli Errori (`src/error.rs`)**:
   - Creato l'enum `GlinerError` con codifica rigorosa (es. `E_GLI_001` per OOM Device Binding, `E_GLI_002` per OOM Standard, `E_GLI_004` per Binding non supportato).
2. **Supervisor e Fallback Dinamico (`src/lib.rs`)**:
   - Creato l'enum di stato `ExecutionMode` (`IoBinding` e `Standard`).
   - Sdoppiato il blocco di inferenza in `extract_iobinding()` e `extract_standard()`.
   - Il metodo principale `extract()` agisce ora da Smart Wrapper: tenta l'esecuzione VRAM-nativa e, in caso di OOM o mancato supporto hardware, commuta lo stato atomico (RwLock) per fare fallback silenzioso alla versione standard, garantendo robustezza totale.
3. **Miglioramento Strumenti di Test (`examples/test_benchmark.rs`)**:
   - Introdotto un log progressivo ogni 10 esecuzioni per monitorare l'avanzamento dei benchmark senza sembrare bloccato.
   - Validata la baseline in `--release` mode (confermando la velocità di ~157ms in Modalità Standard).

---

## 🚧 Il Blocco Architetturale Attuale (Perché l'IOBinding è in sospeso)
Attualmente, l'output di un layer ONNX non può essere agganciato "così com'è" al layer successivo. L'implementazione attuale in Rust (derivata dalla pipeline PyTorch originale) estrae i tensori dalla VRAM per manipolarli sulla CPU tramite `ndarray` usando cicli `for`:
- Slicing dei token tramite `word_to_token_maps` per creare `text_embs`.
- Estrazione del singolo token (`prompt_tok_idx`) per creare `pc_emb_first`.
- Calcolo del valore massimo (`ArgMax`) dell'output di CountPred per ricavare il numero intero `pred_count`.
- Moltiplicazione finale di similitudine (Einsum) e softmax effettuata nativamente in Rust.

**Conclusione:** Finché queste operazioni matematiche e di slicing vengono eseguite su CPU dall'host (Rust), è tecnicamente impossibile mantenere un flusso di memoria contiguo in VRAM.

---

## 📝 Fase 2: Da Fare (Esportazione ONNX Intelligente - Python)
Per abilitare l'IOBinding, le operazioni di manipolazione attualmente fatte in Rust devono essere "fuse" all'interno dei modelli ONNX usando lo script di esportazione (`export_gliner2_onnx.py`). Questo garantisce accelerazione hardware automatica e indipendenza dall'architettura (CUDA, CoreML, NPU).

1. **Fondere lo Slicing in ONNX (`Gather`)**:
   - Modificare il layer che genera `text_embs` affinché accetti direttamente l'array `word_to_token_maps` come input e utilizzi l'operatore ONNX `Gather` per estrarre i sub-token rimanendo in VRAM.
2. **Fondere ArgMax in CountPred**:
   - Modificare `count_pred` in Python affinché l'ultimo nodo sia un `torch.argmax()`. In questo modo ONNX restituirà direttamente l'intero (`pred_count`) calcolato dalla GPU.
3. **Fondere Einsum/Similitudine**:
   - Includere la moltiplicazione tensoriale (Dot Product) tra le `span_embeddings` e le proiezioni dello schema all'interno della rete neurale, restituendo le probabilità finali anziché i tensori latenti grezzi.

---

## 📝 Fase 3: Da Fare (Implementazione Rust IOBinding - `src/lib.rs`)
Una volta che i modelli ONNX saranno capaci di elaborare tutto autonomamente:

1. **Inizializzazione Memory Allocator**:
   - Usare `ort::memory::MemoryInfo` per individuare l'allocatore hardware attivo (es. CUDA per NVIDIA, CoreML per Mac).
2. **Creazione dei Binding**:
   - Implementare la logica in `extract_iobinding()`:
     ```rust
     let mut binding = session.create_binding()?;
     binding.bind_input("input_name", ...)?;
     binding.bind_output_to_device("output_name", &memory_info)?;
     session.run_with_binding(&mut binding)?;
     ```
3. **Passaggio dei Puntatori (`Value`)**:
   - Prelevare gli output generati in VRAM (come `Value` pointer) e assegnarli direttamente come input al binding della sessione successiva (`bind_input`), completando la catena Zero-Copy.
4. **Gestione Errori Hardware**:
   - Catturare eventuali incompatibilità del driver (se il provider non supporta IOBinding) e lanciare l'errore codificato `E_GLI_004` (`BindingNotSupported`) per innescare automaticamente il fallback a Modalità 1.
