
#!/bin/bash

for dir in */ ; do 
	#echo $dir
	dir_ch=`echo $dir | cut -d '/' -f 1`
	echo $dir_ch
	counter=1
	for file in $dir*; do
		
		#if [ ! -f $dir$dir_ch$counter.png ]; then
			mv -f "$file" $dir$dir_ch"_"$counter.png
		#fi

		((counter+=1))
	done
done


#
#  find ./ -type f -print | sort -zR | tail -n +2000 |xargs rm