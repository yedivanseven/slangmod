FILES = system("ls -1 {folder}/{subdir}/*.csv")
LABELS = system("ls -1 {folder}/{subdir}/*.csv | xargs -n 1 basename | sed -e 's/.txt//'")

set datafile separator comma
set title "Convergence"
set xlabel "# batches"
set ylabel "training loss"

plot for [i=1:words(FILES)] word(FILES,i) using 0:1 with lines title word(LABELS,i)
