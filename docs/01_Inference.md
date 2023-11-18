# :closed_book: Inference

This document contains instructions for running the READMem variants (*i.e.,* MiVOS, STCN, QDMN) on the DAVIS (D17) and Long-Video (LV1) datasets.

***Prerequisites:***
- A set workspace as described in *[00_Installation.md](Installation.md)*
- :hotsprings: The D17 and LV1 datasets downloaded.

---

## 🟥 READMem-MiVOS <a name="MiVOS"></a>
- Most useful arguments IMO. But use the ```python READMem_MiVOS.py -h``` for more helpful details.
  ```bash
  python READMem_MiVOS.py \
      --model [path/to/the/MiVOS/saves/propagation_model.pth] \
      --output [output/folder/name] \
      --dataset [dataset_name (either D17 or LV1)] \
      --split [by default on val] \
      --mem_confi [path/to/the/memory/configuration] \
      --silence [to avoid icecream statements]
  ```
- A working command from the get-go, if everything is set correctly:
  ```bash
  python Test_READMem/READMem_MiVOS.py \
      --output OUTPUT_TEST \
      --dataset D17 \
      --silence
  ```

---

## 🟦 READMem-STCN <a name="STCN"></a>
- General usage: (use the ```-h``` argument for more details)
  ```bash
  python READMem_STCN.py \
      --model [path/to/the/STCN/saves/STCN.pth] \
      --output [output/folder/name] \
      --dataset [dataset_name (either D17 or LV1)] \
      --split [by default on val] \
      --mem_confi [path/to/the/memory/configuration] \
      --silence [to avoid icecream statements]
  ```
- Command from the get-go:
  ```bash
  python Test_READMem/READMem_STCN.py \
      --output OUTPUT_TEST_STCN \
      --dataset D17 \
      --silence
  ```

---

## 🟧 READMem-QDMN <a name="QDMN"></a>
- General usage: (use the ```-h``` argument for more details)
  ```bash
  python READMem_QDMN.py \
      --model [path/to/the/QDMN/saves/QDMN.pth] \
      --output [output/folder/name] \
      --dataset [dataset_name (either D17 or LV1)] \
      --split [by default on val] \
      --mem_confi [path/to/the/memory/configuration] \
      --silence [to avoid icecream statements]
  ```
- Command from the get-go:
  ```bash
  python Test_READMem/READMem_QDMN.py \
      --output OUTPUT_TEST_QDMN \
      --dataset D17 \
      --silence
  ```



