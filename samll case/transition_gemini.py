import matplotlib.pyplot as plt
import numpy as np

# Define the energy levels and their potentials
# (Shifted so that 'ATTITUDE' = 0)
levels = {
    'ATTITUDE': 0.0,
    'DISCIPLINE': 0.5,
    'EXCELLENT': 4.8,
    'BLISSFUL': 5.2,
    'GRUMPY': 20.3,
    'TURKEY': 21.8,
    'TENSELY': 24.6,
    'BOYCOTT': 25.0,
    'ACCUMULATE': 27.6,
    'SCUTTLE': 28.4,
    'QUARTER': 28.5,
    'FLURRY': 42.7,
    'BUZZY': 57.5,
}

level_names = list(levels.keys())
level_energies = list(levels.values())

# Transition data N(f->g)
transitions_counts = {
    ('ACCUMULATE', 'ATTITUDE'): 565,
    ('BLISSFUL', 'ATTITUDE'): 353,
    ('BOYCOTT', 'ATTITUDE'): 969,
    ('BUZZY', 'ATTITUDE'): 142,
    ('DISCIPLINE', 'ATTITUDE'): 954,
    ('EXCELLENT', 'ATTITUDE'): 822,
    ('FLURRY', 'ATTITUDE'): 324,
    ('GRUMPY', 'ATTITUDE'): 949,
    ('QUARTER', 'ATTITUDE'): 812,
    ('SCUTTLE', 'ATTITUDE'): 801,
    ('TENSELY', 'ATTITUDE'): 150,
    ('TURKEY', 'ATTITUDE'): 620,
    ('ATTITUDE', 'DISCIPLINE'): 564,
    ('ATTITUDE', 'EXCELLENT'): 7,
    ('BUZZY', 'EXCELLENT'): 12,
    ('ATTITUDE', 'BLISSFUL'): 2,
    ('BUZZY', 'BLISSFUL'): 1,
    ('BUZZY', 'GRUMPY'): 3,
    ('FLURRY', 'GRUMPY'): 3,
    ('FLURRY', 'TURKEY'): 2,
    ('BUZZY', 'TENSELY'): 1,
    ('BUZZY', 'BOYCOTT'): 7,
    ('BUZZY', 'ACCUMULATE'): 1,
    ('BUZZY', 'SCUTTLE'): 1,
    ('BUZZY', 'QUARTER'): 1,
    ('BUZZY', 'FLURRY'): 3,
}

# Calculate N0 as specified: N0 = 20000/13
N0 = 20000 / 13

# Calculate transition probabilities T(f,g) = min(N(f->g) * N0, 1)
transitions_prob = {k: min(v / N0, 1) for k, v in transitions_counts.items()}

# Setup the plot with 2 subplots
# ax1: top (all high energy levels shown as ~infinity) - 25%
# ax2: bottom (ATTITUDE, DISCIPLINE, EXCELLENT, BLISSFUL) - 75%
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4.5, 3.0), 
                                gridspec_kw={'height_ratios': [1, 3]})
fig.subplots_adjust(hspace=0.05)

# Set y-limits for each subplot - expand to separate 4.8 and 5.2
ax2.set_ylim(-0.5, 7.0)  # ATTITUDE, DISCIPLINE, EXCELLENT, BLISSFUL
ax1.set_ylim(19, 59)      # All other levels (GRUMPY through BUZZY)

# Hide spines between subplots
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# Draw break marks
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Determine which subplot each level belongs to
# Lower levels: ATTITUDE, DISCIPLINE, EXCELLENT, BLISSFUL
# Upper levels: all others
lower_level_names = ['ATTITUDE', 'DISCIPLINE', 'EXCELLENT', 'BLISSFUL']

def get_subplot(name, energy):
    if name in lower_level_names:
        return ax2
    else:
        return ax1

# Draw levels without labels
for name, energy in levels.items():
    ax = get_subplot(name, energy)
    ax.axhline(y=energy, color='k', linestyle='-', linewidth=1.0)
    # Remove state name labels

# Set y-axis labels and ticks
ax2.set_ylabel(r'Potential $\beta V$', fontsize=12)

# Upper subplot - show only one ~infinity tick
ax1.set_yticks([30])  # Middle position
ax1.set_yticklabels([r'$\sim\infty$'], fontsize=12)

# Lower subplot
ax2.set_yticks([levels[name] for name in lower_level_names])
ax2.set_yticklabels([str(levels[name]) for name in lower_level_names], fontsize=10)

# --- Draw arrows for transitions ---
# Group transitions by type for better visualization
transitions_to_attitude = [(f, 'ATTITUDE') for f, g in transitions_counts.keys() if g == 'ATTITUDE']
transitions_from_attitude = [('ATTITUDE', g) for f, g in transitions_counts.keys() if f == 'ATTITUDE']
transitions_buzzy = [(f, g) for f, g in transitions_counts.keys() if f == 'BUZZY' and g != 'ATTITUDE']
transitions_flurry = [(f, g) for f, g in transitions_counts.keys() if f == 'FLURRY']

# Combine and order transitions
ordered_transitions = (
    sorted(transitions_to_attitude, key=lambda x: levels[x[0]]) +
    transitions_from_attitude +
    transitions_buzzy +
    [t for t in transitions_flurry if t not in transitions_buzzy]
)

# Assign x-positions
x_positions = np.linspace(0.1, 0.9, len(ordered_transitions))

for i, (f, g) in enumerate(ordered_transitions):
    prob = transitions_prob.get((f, g))
    if prob is None:
        continue

    y_f = levels[f]
    y_g = levels[g]
    
    arrow_width = 0.3 + 1.5 * prob
    x = x_positions[i]
    
    dy = y_g - y_f
    
    # Determine which subplot each level belongs to
    f_ax = get_subplot(f, y_f)
    g_ax = get_subplot(g, y_g)
    
    # Adjust start and end points for arrows - align to energy levels
    if f_ax == g_ax:
        # Same subplot - arrows start and end at energy levels
        y_start_arrow = y_f
        y_end_arrow = y_g
    else:
        # Cross-axis - arrows start and end at energy levels
        y_start_arrow = y_f
        y_end_arrow = y_g

    # Draw arrows without labels
    if f_ax == g_ax:
        # Within same subplot
        f_ax.annotate("", xy=(x, y_end_arrow), xytext=(x, y_start_arrow),
                     arrowprops=dict(arrowstyle="->", color="k", lw=arrow_width))
    else:
        # Cross-axis transitions
        if f_ax == ax1 and g_ax == ax2:
            # ax1 to ax2
            f_ax.plot([x, x], [y_start_arrow, ax1.get_ylim()[0]], 
                     color='k', lw=arrow_width, clip_on=False, zorder=1)
            g_ax.annotate("", xy=(x, y_end_arrow), xytext=(x, ax2.get_ylim()[1]),
                         arrowprops=dict(arrowstyle="->", color="k", lw=arrow_width),
                         clip_on=False, zorder=1)
        else:  # f_ax == ax2 and g_ax == ax1
            # ax2 to ax1
            f_ax.plot([x, x], [y_start_arrow, ax2.get_ylim()[1]], 
                     color='k', lw=arrow_width, clip_on=False, zorder=1)
            g_ax.annotate("", xy=(x, y_end_arrow), xytext=(x, ax1.get_ylim()[0]),
                         arrowprops=dict(arrowstyle="->", color="k", lw=arrow_width),
                         clip_on=False, zorder=1)
    # Remove transition probability labels


ax1.set_xlim(0, 1.0)
ax2.set_xlim(0, 1.0)
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

plt.tight_layout(pad=0.2)
import os
os.makedirs("./figures/word_gemini", exist_ok=True)
plt.savefig("./figures/word_gemini/transition_diagram_gemini.pdf")

print("Transition diagram saved as transition_diagram_gemini.pdf")
print(f"N0 = {N0:.4f}")
