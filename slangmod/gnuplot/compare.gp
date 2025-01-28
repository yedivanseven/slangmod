FILES = system("ls -1 convergence_*.txt")
LABELS = system("ls -1 convergence_*.txt | sed -e 's/convergence_//' -e 's/.txt//'")

set title "Convergence"
set xlabel "# batches"
set ylabel "training loss"

plot for [i=1:words(FILES)] word(FILES,i) using 0:1 with lines title word(LABELS,i)
