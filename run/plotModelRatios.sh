# This file does only the plots in the paper
# This is so can use custom ylimits to make plots look nice
# The other ones are default

#Where we are running this from
# myDir='/home/ryan/Documents/GitHub/PullRequests_SwanLat/cfuns/tomlWriters/charmBaryon/'
myDir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The python executable
pythonExe=python #/home/ryan/installs/conda/miniconda3/envs/gen/bin/python

# Where modified toml files are
outTomlDir=${myDir}/modifiedToml_Ratios/12_06_2023/ModelRatio/

# the plotting script
scriptDir=${myDir}/../code/bin/
exe=GPlots.py

# Data Dir
dataDir=${myDir}/../

#for temp in cool;
#for temp in mid;
for temp in cool mid;
do
    #for RAT in single;
    #for RAT in double;
    for RAT in single double;
    do
	#for OP in sigma12_3fl_udc;
	#for OP in doublet_2fl_ccs;
	for OP in doublet_2fl_ccs sigma12_3fl_udc;
	do
	    #for PAR in Pos;
	    #for PAR in Neg;
	    for PAR in Pos Neg
	    do
		cd $dataDir
		echo ${pythonExe} ${scriptDir}${exe} ${outTomlDir}/${OP}_${temp}_${RAT}_${PAR}_Ratios.toml
		${pythonExe} ${scriptDir}${exe} ${outTomlDir}/${OP}_${temp}_${RAT}_${PAR}_Ratios.toml
		cd -
	    done
	done
    done
done
