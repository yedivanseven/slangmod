FILE = system("ls convergence_*.txt | tail -n 1")

set title "Convergence"
set xlabel "# batches"
set ylabel "training loss"
set y2label "learning rate"
set ytics nomirror
set y2tics
set style line 1 lc rgbcolor "#1f77b4"
set style line 2 lc rgbcolor "#ff7f0e"

plot FILE using 0:1 with lines axes x1y1 ls 1 title "training loss", \
     FILE using 0:2 with lines axes x1y2 ls 2 title "learning rate"

while (1) {
    replot
    pause 1
}
