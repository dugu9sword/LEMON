#!/bin/bash



for file in task-*.txt; do
	if [[ ! -e "last.$file" ]]; then
		epoch_line=$(grep -n 'EPOCH 29' $file | awk -F ':' '{print $1}')
		total_line=$(wc -l $file | awk '{print $1}')
		#echo $epoch_line
		#echo $total_line
		tail_num=$(( total_line - epoch_line ))
		#echo $tail_num
		tail -n $tail_num $file > "last.$file"
		echo "generate $logpath/last.$file"
	else
		echo "$logpath/last.$file exists"
	fi
done


for file in $( find last.*.txt | sort -n -k 2 -t - ); do
	echo ''
	echo $file
	ll=$(grep 'LOC/LOC' $file|wc -l)
	ln=$(grep 'LOC/[^L]' $file|wc -l)
	nl=$(grep '[^C]/LOC' $file|wc -l)

	pp=$(grep 'PER/PER' $file|wc -l)
	np=$(grep '[^R]/PER' $file|wc -l)
	pn=$(grep 'PER/[^P]' $file|wc -l)

	oo=$(grep 'ORG/ORG' $file|wc -l)
	no=$(grep '[^G]/ORG' $file|wc -l)
	on=$(grep 'ORG/[^O]' $file|wc -l)

	gg=$(grep 'GPE/GPE' $file|wc -l)
	ng=$(grep '[^E]/GPE' $file|wc -l)
	gn=$(grep 'GPE/[^G]' $file|wc -l)

	echo TYP T/T  F/T  T/F
	echo LOC +$ll -$nl -$ln
	echo PER +$pp -$np -$pn
	echo ORG +$oo -$no -$on
	echo GPE +$gg -$ng -$gn

	rec=$( echo "scale=3; ( $ll + $pp + $oo + $gg + 0.0)/( $ll + $pp + $oo + $gg + $nl + $np + $no + $ng )"| bc)
	pre=$( echo "scale=3; ( $ll + $pp + $oo + $gg + 0.0)/( $ll + $pp + $oo + $gg + $ln + $pn + $on + $gn )"| bc)
	f1=$( echo "scale=3; 2 / ( 1 / $pre + 1 / $rec )"| bc)
	echo $pre $rec $f1
done


