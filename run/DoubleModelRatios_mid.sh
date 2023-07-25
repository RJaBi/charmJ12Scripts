

#Where we are running this from
# myDir='/home/ryan/Documents/GitHub/PullRequests_SwanLat/cfuns/tomlWriters/charmBaryon/'
myDir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The python executable
pythonExe=python #/home/ryan/installs/conda/miniconda3/envs/gen/bin/python

# The script used to turn the template into the full files
writeScript=${myDir}/writeModelRatios.py

# Where we are putting all the modified toml files
outTomlDir=${myDir}/AutoToml_Ratios/12_06_2023/ModelRatio/
# Make a folder to put the modified toml in
mkdir -p ${outTomlDir}

# the plotting script
#scriptDir=/home/ryan/Documents/GitHub/PullRequests_SwanLat/cfuns/currentlyUsed/
scriptDir=${myDir}/../code/bin/
exe=GPlots.py

# Data Dir
#dataDir=/home/ryan/Documents/2022/Gen2L/Hawk/charmBaryonData/
#dataDir=/home/ryan/Documents/2023/Gen2L/charmBaryonDataJ1_2/
dataDir=${myDir}/../


# Loop over temperatures - cool, mid, hot
#for temp in mid;
#for temp in cool mid;
for temp in mid;
#for temp in hot;
do
    # The template toml file
    template=${myDir}/template-${temp}-base.toml
    

    # Sigma_c(udc)
    OP=sigma12.3fl
    UN=sigma12_3fl
    qqq=udc
    had='\Sigma_{c}(udc)'
    for RAT in double;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak 's~"None", "None"~0.82, 1.45~g' {} \;  # set ylimits

	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak 's~"None", "None"~0.95, 1.75~g' {} \;  # set ylimits
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # xi^\prime(usc)
    OP=sigma12.3fl
    UN=sigma12_3fl
    qqq=usc
    had='\Xi_{c}^{\prime}(usc)'
    for RAT in double;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # Omega_{c}(ssc)
    OP=doublet.2fl
    UN=doublet_2fl
    qqq=ssc
    had='\Omega_{c}(ssc)'
    for RAT in double;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # Sigma(uus)
    OP=lambda.3fl
    UN=lambda_3fl
    qqq=udc
    had='\Lambda_{c}(udc)'
    for RAT in double;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # xi_{c}(usc
    OP=lambda.3fl
    UN=lambda_3fl
    qqq=usc
    had='\Xi_{c}(usc)'
    for RAT in double;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # xi_{cc}(ccu)
    OP=doublet.2fl
    UN=doublet_2fl
    qqq=ccu
    had='\Xi_{cc}(ccu)'
    for RAT in double;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak 's~"None", "None"~0.875, 1.33~g' {} \;  # set ylimits
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak 's~"None", "None"~0.9, 2.025~g' {} \;  # set ylimits
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # \Omega_cc (ccs)
    OP=doublet.2fl
    UN=doublet_2fl
    qqq=ccs
    had='\Omega_{cc}(ccs)'
    for RAT in double;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak 's~"None", "None"~0.91, 1.275~g' {} \;  # set ylimits
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ModelRatio~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak 's~"None", "None"~0.9, 2.025~g' {} \;  # set ylimits
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
done
