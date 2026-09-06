# Local Qwen3 4B benchmark: Zebrafish ESM Discovery

## Executive summary

This report records one local benchmark run of `qwen3:4b-instruct` (4B, Q4_K_M) through Ollama. The model interpreted each biological question and proposed candidate genes. The existing pipeline then used UniProt/Ensembl for deterministic identity resolution and the private local ESM database for similarity search. Gemini was not called, Google Search grounding was disabled, and embeddings remained local.

- Questions: **24** across **8 categories** (ambiguous, cardiac, crispr, immune, macrophage, neuronal, pigmentation, vascular)
- At least one validated local ESM seed: **23/24 (95.8%)**
- At least one match to the case's predefined canonical reference examples: **17/24 (70.8%)**
- Median end-to-end latency: **25.24 seconds per question**
- Mean end-to-end latency: **36.17 seconds per question**
- Observed latency range: **22.07–102.62 seconds**
- Total runtime: **868.04 seconds**
- Web grounding used: **no**

The canonical-reference check is a transparent heuristic, not a biological gold-standard accuracy score. A reference miss can still contain relevant biology—for example, the general macrophage query returned and validated `c1qa`—while a validated identifier is not by itself proof that every proposed protein is relevant to the question.

## Aggregate case table

| # | Category | Prompt | Reference outcome | Validated seeds | Seconds |
|---:|---|---|---|---:|---:|
| 1 | macrophage | Which proteins mark zebrafish macrophages? | No reference hit; validated seeds produced | 2 | 87.42 |
| 2 | macrophage | Find proteins involved in macrophage phagocytosis. | Reference hit + validated seeds | 5 | 28.39 |
| 3 | macrophage | What genes identify microglia in the zebrafish brain? | Reference hit + validated seeds | 5 | 32.65 |
| 4 | vascular | Which proteins mark vascular endothelial cells in zebrafish? | Reference hit + validated seeds | 7 | 28.15 |
| 5 | vascular | Find core zebrafish angiogenesis proteins. | Reference hit + validated seeds | 7 | 23.64 |
| 6 | vascular | Which genes help form the zebrafish blood-brain barrier? | No reference hit; validated seeds produced | 4 | 23.14 |
| 7 | neuronal | Give me general neuronal marker proteins in zebrafish. | Reference hit + validated seeds | 4 | 28.61 |
| 8 | neuronal | Which genes identify dopaminergic neurons in zebrafish? | Reference hit + validated seeds | 3 | 102.62 |
| 9 | neuronal | Find proteins associated with zebrafish motor neurons. | No reference hit; validated seeds produced | 4 | 25.31 |
| 10 | immune | Which proteins respond to bacterial infection in zebrafish? | No validated seed | 0 | 53.82 |
| 11 | immune | Find zebrafish antiviral interferon-response proteins. | Reference hit + validated seeds | 5 | 26.16 |
| 12 | immune | Which genes mark adaptive immune lymphocytes in zebrafish? | No reference hit; validated seeds produced | 1 | 76.97 |
| 13 | pigmentation | Find proteins required for zebrafish melanophore pigmentation. | Reference hit + validated seeds | 6 | 25.10 |
| 14 | pigmentation | Which genes are important for zebrafish xanthophores? | Reference hit + validated seeds | 8 | 23.59 |
| 15 | pigmentation | What proteins control zebrafish pigment stripe formation? | Reference hit + validated seeds | 6 | 24.78 |
| 16 | cardiac | Which proteins mark zebrafish cardiomyocytes? | Reference hit + validated seeds | 4 | 22.08 |
| 17 | cardiac | Find genes involved in the zebrafish cardiac conduction system. | No reference hit; validated seeds produced | 5 | 24.21 |
| 18 | cardiac | Which proteins contribute to zebrafish heart regeneration? | Reference hit + validated seeds | 6 | 23.98 |
| 19 | crispr | Suggest candidate CRISPR targets to reduce zebrafish pigmentation. | Reference hit + validated seeds | 8 | 24.62 |
| 20 | crispr | What genes could I knock out to reduce macrophage development in zebrafish? | No reference hit; validated seeds produced | 3 | 65.46 |
| 21 | crispr | Suggest zebrafish CRISPR targets for disrupting blood-vessel development. | Reference hit + validated seeds | 7 | 24.31 |
| 22 | ambiguous | What proteins make the fish transparent? | Reference hit + validated seeds | 5 | 25.18 |
| 23 | ambiguous | Which cells-eating-debris proteins matter after a zebrafish brain injury? | Reference hit + validated seeds | 4 | 25.78 |
| 24 | ambiguous | Find genes that make vessels grow around a wound. | Reference hit + validated seeds | 8 | 22.07 |

## Detailed results

### 1. macrophage: Which proteins mark zebrafish macrophages?

- Outcome: **No reference hit; validated seeds produced**
- Predefined reference examples: `mpeg1`, `mpeg1.1`, `mfap4`, `csf1ra`
- Model-proposed zebrafish genes: `ms4a1a`, `ms4a1b`, `c1qa`, `tlr4`, `ccl2`, `ccl3`, `ccl12`, `lcn2`, `s100a8`
- Reference hits: None
- Deterministically validated seed genes: `c1qa`, `ccl38a.5`
- Top ESM neighbors: `c1qb` (0.96823; closest seed `c1qa`), `c1qc` (0.96774; closest seed `c1qa`), `ccl39.6` (0.95149; closest seed `ccl38a.5`), `col6a4a` (0.93034; closest seed `c1qa`), `ccl38.1` (0.92935; closest seed `ccl38a.5`)
- End-to-end latency: 87.42 seconds

### 2. macrophage: Find proteins involved in macrophage phagocytosis.

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `mpeg1`, `mpeg1.1`, `marco`, `spi1b`, `lcp1`
- Model-proposed zebrafish genes: `mpeg1`, `c1qb`, `cd36`, `actb`, `rabs11`, `syk`, `caspase-3`, `tmem119`, `dscam`, `c1r`
- Reference hits: `mpeg1`, `mpeg1.1`
- Deterministically validated seed genes: `mpeg1.1`, `c1qb`, `cd36`, `dscama`, `c1r`
- Top ESM neighbors: `dscamb` (0.99638; closest seed `dscama`), `dscaml1` (0.98725; closest seed `dscama`), `mpeg1.2` (0.98521; closest seed `mpeg1.1`), `c1qa` (0.96823; closest seed `c1qb`), `fras1` (0.96653; closest seed `dscama`)
- End-to-end latency: 28.39 seconds

### 3. macrophage: What genes identify microglia in the zebrafish brain?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `apoeb`, `p2ry12`, `csf1ra`, `mpeg1.1`
- Model-proposed zebrafish genes: `pf4`, `mpeg1`, `tmem119`, `c1qtb`, `cd68`, `lcn2`, `s100a9`, `c1qa`, `c1qb`, `c1qc`
- Reference hits: `mpeg1.1`
- Deterministically validated seed genes: `mpeg1.1`, `cd68`, `c1qa`, `c1qb`, `c1qc`
- Top ESM neighbors: `mpeg1.2` (0.98521; closest seed `mpeg1.1`), `sort1` (0.95033; closest seed `mpeg1.1`), `elapor2b` (0.94653; closest seed `mpeg1.1`), `selenon` (0.94548; closest seed `mpeg1.1`), `elapor1` (0.94382; closest seed `mpeg1.1`)
- End-to-end latency: 32.65 seconds

### 4. vascular: Which proteins mark vascular endothelial cells in zebrafish?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `kdrl`, `cdh5`, `fli1a`, `etv2`, `pecam1`
- Model-proposed zebrafish genes: `kdrl`, `vegfr2`, `flt1`, `ephrinb2`, `tie2`, `s1pr1`, `mmp9`, `eng`, `pdpn`
- Reference hits: `kdrl`
- Deterministically validated seed genes: `kdrl`, `flt1`, `tek`, `s1pr1`, `mmp9`, `eng`, `si:ch211-156j16.1`
- Top ESM neighbors: `flt4` (0.98341; closest seed `kdrl`), `kdr` (0.98148; closest seed `kdrl`), `s1pr3b` (0.97997; closest seed `s1pr1`), `kita` (0.97515; closest seed `kdrl`), `kitb` (0.97478; closest seed `kdrl`)
- End-to-end latency: 28.15 seconds

### 5. vascular: Find core zebrafish angiogenesis proteins.

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `vegfaa`, `kdrl`, `kdr`, `flt1`, `dll4`
- Model-proposed zebrafish genes: `kdrl`, `vegfa`, `bmp4`, `notch1`, `flt1`, `eng`, `sdf1`, `ephrinb2`, `tie2`, `mmp9`
- Reference hits: `flt1`, `kdrl`, `vegfaa`
- Deterministically validated seed genes: `kdrl`, `vegfaa`, `bmp4`, `flt1`, `eng`, `tek`, `mmp9`
- Top ESM neighbors: `flt4` (0.98341; closest seed `kdrl`), `kdr` (0.98148; closest seed `kdrl`), `bmp2b` (0.98114; closest seed `bmp4`), `kita` (0.97515; closest seed `kdrl`), `kitb` (0.97478; closest seed `kdrl`)
- End-to-end latency: 23.64 seconds

### 6. vascular: Which genes help form the zebrafish blood-brain barrier?

- Outcome: **No reference hit; validated seeds produced**
- Predefined reference examples: `cldn5a`, `slc2a1a`, `mfsd2aa`, `cdh5`
- Model-proposed zebrafish genes: `cldn11a`, `cldn11b`, `cldn3`, `tjp1`, `pdpn`, `s100a6`, `kcnj10`, `abca1`, `ephrinb2`
- Reference hits: None
- Deterministically validated seed genes: `cldn11a`, `cldn11b`, `si:ch211-156j16.1`, `abca1a`
- Top ESM neighbors: `abca1b` (0.99241; closest seed `abca1a`), `slc12a7b` (0.95752; closest seed `abca1a`), `atp9b` (0.95711; closest seed `abca1a`), `slc12a2` (0.95573; closest seed `abca1a`), `abca3b` (0.95507; closest seed `abca1a`)
- End-to-end latency: 23.14 seconds

### 7. neuronal: Give me general neuronal marker proteins in zebrafish.

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `elavl3`, `tubb5`, `neurod1`, `map2`
- Model-proposed zebrafish genes: `synapsin1a`, `map2`, `neuromodulin`, `vglut1a`, `gad1a`, `tubulin beta 2a`, `syt1a`, `pax6a`, `neurod1a`
- Reference hits: `map2`
- Deterministically validated seed genes: `map2`, `gad1a`, `syt1a`, `pax6a`
- Top ESM neighbors: `gad1b` (0.99289; closest seed `gad1a`), `pax6b` (0.99054; closest seed `pax6a`), `maptb` (0.96744; closest seed `map2`), `pax7a` (0.96659; closest seed `pax6a`), `map1b` (0.96398; closest seed `map2`)
- End-to-end latency: 28.61 seconds

### 8. neuronal: Which genes identify dopaminergic neurons in zebrafish?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `th`, `slc6a3`, `ddc`, `nr4a2a`
- Model-proposed zebrafish genes: `th`, `dat`, `pmr1`, `drd1`, `drd2`, `tyrosinehydroxylase`, `dop1`, `dop2`, `dop3`, `dop4`
- Reference hits: `slc6a3`, `th`
- Deterministically validated seed genes: `th`, `slc6a3`, `drd1b`
- Top ESM neighbors: `LOC563567` (0.97776; closest seed `drd1b`), `slc5a10` (0.96285; closest seed `slc6a3`), `drd6b` (0.96218; closest seed `drd1b`), `htr1ab` (0.96195; closest seed `drd1b`), `slc6a9` (0.96001; closest seed `slc6a3`)
- End-to-end latency: 102.62 seconds

### 9. neuronal: Find proteins associated with zebrafish motor neurons.

- Outcome: **No reference hit; validated seeds produced**
- Predefined reference examples: `mnx1`, `isl1`, `isl2a`, `chat`, `slc18a3a`
- Model-proposed zebrafish genes: `dmd`, `dach1`, `hcn1`, `kcnq2`, `neurod1`, `tubb3`, `syt1`, `gria1`, `drgn`, `sox11`
- Reference hits: None
- Deterministically validated seed genes: `dmd`, `hcn1`, `syt1a`, `gria1b`
- Top ESM neighbors: `gria1a` (0.99302; closest seed `gria1b`), `gria3b` (0.98849; closest seed `gria1b`), `gria3a` (0.98773; closest seed `gria1b`), `gria4a` (0.98473; closest seed `gria1b`), `gria2b` (0.97925; closest seed `gria1b`)
- End-to-end latency: 25.31 seconds

### 10. immune: Which proteins respond to bacterial infection in zebrafish?

- Outcome: **No validated seed**
- Predefined reference examples: `il1b`, `tnfa`, `mpx`, `lyz`, `nfkbiaa`
- Model-proposed zebrafish genes: `toll_like_receptor_4a`, `toll_like_receptor_6b`, `interleukin_1beta`, `interleukin_10a`, `defensin_1a`, `complement_factor_c3`, `nfrp1a`, `cxcl10a`, `tumor_necrosis_factor_alphaa`, `s100a9a`
- Reference hits: None
- Deterministically validated seed genes: None
- Top ESM neighbors: None
- End-to-end latency: 53.82 seconds

### 11. immune: Find zebrafish antiviral interferon-response proteins.

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `stat1a`, `stat1b`, `mxa`, `isg15`, `irf7`
- Model-proposed zebrafish genes: `ifnra`, `ifngr1`, `isp1`, `rsad2`, `mx1`, `isg15`, `olig1`, `trif`, `myd88`, `irf7`
- Reference hits: `irf7`, `isg15`
- Deterministically validated seed genes: `ifngr1`, `rsad2`, `olig1`, `ticam1`, `myd88`
- Top ESM neighbors: `ifngr1l` (0.97836; closest seed `ifngr1`), `crfb2` (0.97146; closest seed `ifngr1`), `il13ra1` (0.96875; closest seed `ifngr1`), `csf2rb` (0.96870; closest seed `ifngr1`), `ifngr2` (0.96854; closest seed `ifngr1`)
- End-to-end latency: 26.16 seconds

### 12. immune: Which genes mark adaptive immune lymphocytes in zebrafish?

- Outcome: **No reference hit; validated seeds produced**
- Predefined reference examples: `rag1`, `rag2`, `cd4-1`, `cd8a`, `ighm`
- Model-proposed zebrafish genes: `caspase3a`, `il4`, `il10`, `tcralpha`, `tcrbeta`, `igl1`, `igl2`, `lyz1`, `cd4`, `cd8`
- Reference hits: None
- Deterministically validated seed genes: `il10`
- Top ESM neighbors: `gbgt1l5` (0.93324; closest seed `il10`), `dpep2` (0.93172; closest seed `il10`), `si:ch73-380l3.2` (0.92813; closest seed `il10`), `exoc3l4` (0.92776; closest seed `il10`), `pkd2l1` (0.92762; closest seed `il10`)
- End-to-end latency: 76.97 seconds

### 13. pigmentation: Find proteins required for zebrafish melanophore pigmentation.

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `mitfa`, `tyr`, `dct`, `tyrp1b`
- Model-proposed zebrafish genes: `mc1r`, `tyr`, `tyr2`, `dct`, `mitf`, `slc4a10`, `pax7`, `sox10`, `kit`, `caspase3`
- Reference hits: `dct`, `tyr`
- Deterministically validated seed genes: `mc1r`, `tyr`, `dct`, `pax7a`, `sox10`, `kita`
- Top ESM neighbors: `pax7b` (0.99140; closest seed `pax7a`), `kitb` (0.98504; closest seed `kita`), `csf1r` (0.98282; closest seed `kita`), `flt4` (0.98204; closest seed `kita`), `pdgfra` (0.97909; closest seed `kita`)
- End-to-end latency: 25.10 seconds

### 14. pigmentation: Which genes are important for zebrafish xanthophores?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `csf1ra`, `pax7a`, `gch2`
- Model-proposed zebrafish genes: `mc1r`, `tyr`, `tyr2`, `dct`, `slc4a10`, `pax7`, `mitfa`, `cyp1b1`, `kit`, `sox10`
- Reference hits: `pax7a`
- Deterministically validated seed genes: `mc1r`, `tyr`, `dct`, `pax7a`, `mitfa`, `cyp1b1`, `kita`, `sox10`
- Top ESM neighbors: `pax7b` (0.99140; closest seed `pax7a`), `kitb` (0.98504; closest seed `kita`), `csf1r` (0.98282; closest seed `kita`), `cyp1a` (0.98269; closest seed `cyp1b1`), `flt4` (0.98204; closest seed `kita`)
- End-to-end latency: 23.59 seconds

### 15. pigmentation: What proteins control zebrafish pigment stripe formation?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `kita`, `kitlga`, `ednrb1a`, `gja5b`
- Model-proposed zebrafish genes: `mitfa`, `kit`, `tyrosinase`, `slc4a10`, `caspase-3`, `bmp4`, `wnt11`, `engrailed`, `sox10`, `pax7`
- Reference hits: `kita`
- Deterministically validated seed genes: `mitfa`, `kita`, `bmp4`, `wnt11`, `sox10`, `pax7a`
- Top ESM neighbors: `pax7b` (0.99140; closest seed `pax7a`), `kitb` (0.98504; closest seed `kita`), `csf1r` (0.98282; closest seed `kita`), `flt4` (0.98204; closest seed `kita`), `mitfb` (0.98204; closest seed `mitfa`)
- End-to-end latency: 24.78 seconds

### 16. cardiac: Which proteins mark zebrafish cardiomyocytes?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `myl7`, `nkx2.5`, `tnnt2a`, `myh6`
- Model-proposed zebrafish genes: `cmlc1`, `tnnt2a`, `acta2`, `myh7a`, `myl7`, `tropomysin`, `sarcomeric alpha-actin`, `cnn1`, `myl3`
- Reference hits: `myl7`, `tnnt2a`
- Deterministically validated seed genes: `cmlc1`, `acta2`, `myl7`, `myl3`
- Top ESM neighbors: `actc1b` (0.99777; closest seed `acta2`), `acta1a` (0.99651; closest seed `acta2`), `actc2` (0.99640; closest seed `acta2`), `LOC141385692` (0.98430; closest seed `cmlc1`), `acta1b` (0.98210; closest seed `acta2`)
- End-to-end latency: 22.08 seconds

### 17. cardiac: Find genes involved in the zebrafish cardiac conduction system.

- Outcome: **No reference hit; validated seeds produced**
- Predefined reference examples: `hcn4`, `nkx2.5`, `isl1`, `tbx3`
- Model-proposed zebrafish genes: `ttn`, `scn5a`, `cav1.2`, `kcnh2`, `kcnq1`, `gja1`, `hcn1`, `ryr2`, `cacna1c`, `tmem16a`
- Reference hits: None
- Deterministically validated seed genes: `kcnh6a`, `kcnq1.1`, `gja1`, `hcn1`, `cacna1c`
- Top ESM neighbors: `LOC101883600` (0.98673; closest seed `kcnh6a`), `cacna1fb` (0.98052; closest seed `cacna1c`), `cacna1aa` (0.97892; closest seed `cacna1c`), `kcnh7` (0.97785; closest seed `kcnh6a`), `kcnq4` (0.96911; closest seed `cacna1c`)
- End-to-end latency: 24.21 seconds

### 18. cardiac: Which proteins contribute to zebrafish heart regeneration?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `gata4`, `hand2`, `stat3`, `nrg1`
- Model-proposed zebrafish genes: `casp9a`, `mef2c`, `gata4`, `hand2`, `tcf22`, `myh7a`, `acta2`, `bmp4`, `notch1a`, `foxo1a`
- Reference hits: `gata4`, `hand2`
- Deterministically validated seed genes: `mef2ca`, `hand2`, `acta2`, `bmp4`, `notch1a`, `foxo1a`
- Top ESM neighbors: `actc1b` (0.99777; closest seed `acta2`), `acta1a` (0.99651; closest seed `acta2`), `actc2` (0.99640; closest seed `acta2`), `notch2` (0.98777; closest seed `notch1a`), `notch1b` (0.98672; closest seed `notch1a`)
- End-to-end latency: 23.98 seconds

### 19. crispr: Suggest candidate CRISPR targets to reduce zebrafish pigmentation.

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `mitfa`, `tyr`, `slc24a5`
- Model-proposed zebrafish genes: `mitfa`, `tyr`, `tyr2`, `mc1r`, `slc4a10`, `dct`, `slc24a4`, `pax7`, `sox10`, `kit`
- Reference hits: `mitfa`, `tyr`
- Deterministically validated seed genes: `mitfa`, `tyr`, `mc1r`, `dct`, `slc23a4`, `pax7a`, `sox10`, `kita`
- Top ESM neighbors: `pax7b` (0.99140; closest seed `pax7a`), `kitb` (0.98504; closest seed `kita`), `csf1r` (0.98282; closest seed `kita`), `flt4` (0.98204; closest seed `kita`), `mitfb` (0.98204; closest seed `mitfa`)
- End-to-end latency: 24.62 seconds

### 20. crispr: What genes could I knock out to reduce macrophage development in zebrafish?

- Outcome: **No reference hit; validated seeds produced**
- Predefined reference examples: `irf8`, `csf1ra`, `spi1b`
- Model-proposed zebrafish genes: `cish`, `ccl3`, `ccl2`, `ccl7`, `ccl12`, `spdef`, `lcn2`, `tcf7l2`, `cish.1`, `ccl2b`
- Reference hits: None
- Deterministically validated seed genes: `cisha`, `ccl38a.5`, `spdef`
- Top ESM neighbors: `cishb` (0.96892; closest seed `cisha`), `ccl39.6` (0.95149; closest seed `ccl38a.5`), `limk2` (0.94959; closest seed `cisha`), `sh2d3cb` (0.94854; closest seed `cisha`), `socs1b` (0.94664; closest seed `cisha`)
- End-to-end latency: 65.46 seconds

### 21. crispr: Suggest zebrafish CRISPR targets for disrupting blood-vessel development.

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `kdrl`, `vegfaa`, `etv2`, `fli1a`
- Model-proposed zebrafish genes: `kdr`, `vegfa`, `notch1`, `bmp4`, `tie2`, `eng`, `sdf1a`, `flt1`, `ephrinb2`, `pdk1`
- Reference hits: `kdrl`, `vegfaa`
- Deterministically validated seed genes: `kdrl`, `vegfaa`, `bmp4`, `tek`, `eng`, `flt1`, `pdk1`
- Top ESM neighbors: `pdk4` (0.98664; closest seed `pdk1`), `pdk2a` (0.98635; closest seed `pdk1`), `flt4` (0.98341; closest seed `kdrl`), `kdr` (0.98148; closest seed `kdrl`), `bmp2b` (0.98114; closest seed `bmp4`)
- End-to-end latency: 24.31 seconds

### 22. ambiguous: What proteins make the fish transparent?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `mitfa`, `tyr`, `slc24a5`, `mpv17`
- Model-proposed zebrafish genes: `crb1`, `pax6a`, `sox10a`, `mitfa`, `tcf7l2`, `krt8a`, `gata4a`, `cldn11a`, `s100a9a`, `fli1a`
- Reference hits: `mitfa`
- Deterministically validated seed genes: `crb1`, `pax6a`, `mitfa`, `cldn11a`, `fli1`
- Top ESM neighbors: `pax6b` (0.99054; closest seed `pax6a`), `mitfb` (0.98204; closest seed `mitfa`), `fli1rs` (0.97676; closest seed `fli1`), `crb2a` (0.97576; closest seed `crb1`), `eys` (0.97515; closest seed `crb1`)
- End-to-end latency: 25.18 seconds

### 23. ambiguous: Which cells-eating-debris proteins matter after a zebrafish brain injury?

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `mpeg1.1`, `apoeb`, `csf1ra`, `p2ry12`
- Model-proposed zebrafish genes: `mpeg1`, `mpeg2`, `tmem119`, `cd68`, `s100a9`, `c1qa`, `lcn2`, `tmem116a`, `mpeg1a`, `c1qb`
- Reference hits: `mpeg1.1`
- Deterministically validated seed genes: `mpeg1.1`, `cd68`, `c1qa`, `c1qb`
- Top ESM neighbors: `mpeg1.2` (0.98521; closest seed `mpeg1.1`), `c1qc` (0.96774; closest seed `c1qa`), `sort1` (0.95033; closest seed `mpeg1.1`), `elapor2b` (0.94653; closest seed `mpeg1.1`), `selenon` (0.94548; closest seed `mpeg1.1`)
- End-to-end latency: 25.78 seconds

### 24. ambiguous: Find genes that make vessels grow around a wound.

- Outcome: **Reference hit + validated seeds**
- Predefined reference examples: `vegfaa`, `kdrl`, `flt1`, `dll4`
- Model-proposed zebrafish genes: `spry2`, `vegfa`, `kdrl`, `flt1`, `bmp2b`, `eng2`, `dpp4`, `mmp9`, `timp1`, `pdpn`
- Reference hits: `flt1`, `kdrl`, `vegfaa`
- Deterministically validated seed genes: `spry2`, `vegfaa`, `kdrl`, `flt1`, `bmp2b`, `eng2a`, `mmp9`, `si:ch211-156j16.1`
- Top ESM neighbors: `eng2b` (0.98701; closest seed `eng2a`), `flt4` (0.98341; closest seed `kdrl`), `kdr` (0.98148; closest seed `kdrl`), `bmp4` (0.98114; closest seed `bmp2b`), `bmp2a` (0.97812; closest seed `bmp2b`)
- End-to-end latency: 22.07 seconds

## Interpretation for CV and README use

The benchmark supports saying that a local 4B model was integrated and evaluated as a bounded biological-query interpreter. It also supports reporting the exact 24-question seed-resolution and reference-overlap results above. It does not support a claim that the model achieved 95.8% biological accuracy, matched or outperformed Gemini, or operated fully offline: UniProt and Ensembl were still used for public identifier resolution.

A conservative CV bullet:

> Integrated a local Qwen3 4B model through Ollama into a zebrafish protein-discovery pipeline; benchmarked 24 natural-language questions across eight biological categories, producing deterministically validated ESM search seeds for 23/24 cases with 25.25-second median end-to-end latency.

A more technical project-description sentence:

> Added a configurable local Ollama interpretation path while retaining deterministic UniProt/Ensembl identity validation and local ESM similarity search; in one 24-question benchmark, 23 prompts produced validated zebrafish seeds and 17 included at least one predefined canonical reference gene.

## Handoff prompt for another GPT

```text
Use this benchmark report and the repository README to propose factual CV and public README updates.

Requirements:
- State that qwen3:4b-instruct ran locally through Ollama for biological candidate generation.
- State that UniProt/Ensembl validation remained external and deterministic, while ESM embeddings/search stayed local.
- You may report 23/24 prompts producing validated seeds, 17/24 overlapping the predefined reference examples, and 25.25-second median latency.
- Describe the 17/24 measure as reference-example overlap, not accuracy.
- Do not claim comparison, parity, or superiority versus Gemini; no paired Gemini run was performed.
- Mention that exact zebrafish symbol generation was the main observed weakness.
- Keep the CV bullet concise and make the README methodology reproducible.
```
