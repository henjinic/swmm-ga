# Genetic Algorithm on Landuse Map
* Multi-objectives including runoff (caculated by PCSWMM)
* Rule customization

## How to Run
1. Open the PCSWMM project file.
2. Open the script window.
3. Import all the scripts in the *src* folder.
4. Set `SUB_OG_PATH` in the *main.py* to the path to the *sub_og.csv*
5. Run the *main.py*

## Revert to Original Subcatchment
1. Set `ORIGINAL_PATH` in the *swmm.py* to the path to the *original.csv*
2. Uncomment `load(ORIGINAL_PATH)`