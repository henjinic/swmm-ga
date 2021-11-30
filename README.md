# Genetic Algorithm on Landuse Map
* Multi-objectives including runoff (caculated by PCSWMM)
* Rule customization

## How to Run
1. Open the PCSWMM project file.
2. Open script window.
3. Import all the scripts in *src* folder without ```__init__.py``` and *utils* folder.
4. Set `SUB_OG_PATH` in *main.py* to path to *sub_og.csv*
5. Run *main.py*

## Revert to Original Subcatchment
1. Set `ORIGINAL_PATH` in *swmm.py* to path to *original.csv*
2. Uncomment `load(ORIGINAL_PATH)`