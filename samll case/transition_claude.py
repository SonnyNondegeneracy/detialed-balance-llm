import matplotlib.pyplot as plt
import numpy as np

# Define the energy levels and their potentials
levels = {
    'ATTITUDE': 0.0,
    'PERSONAL': 4.1,
    'PROBLEMS': 5.2,
    'TURKEY': 18.2,
    'BUZZY': 34.0,
}

level_names = list(levels.keys())
level_energies = list(levels.values())

# Transition data N(f->g)
transitions_counts = {
    ('BUZZY', 'ATTITUDE'): 3859,
    ('PERSONAL', 'ATTITUDE'): 3879,
    ('PROBLEMS', 'ATTITUDE'): 3558,
    ('TURKEY', 'ATTITUDE'): 4122,
    ('ATTITUDE', 'PERSONAL'): 66,
    ('ATTITUDE', 'PROBLEMS'): 20,
    ('BUZZY', 'TURKEY'): 238,
}

# Calculate transition probabilities T(f,g) = min(N(f->g)/4000, 1)
transitions_prob = {k: min(v / 4000, 1) for k, v in transitions_counts.items()}

# Setup the plot for a PRL single-column figure
# Make it wider and shorter
# ax1 is upper (TURKEY and BUZZY), ax2 is lower (ATTITUDE, PERSONAL, PROBLEMS)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4.5, 2.5), gridspec_kw={'height_ratios': [2, 3]})
fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# Plot the lower levels on ax2 (ATTITUDE, PERSONAL, PROBLEMS)
ax2.set_ylim(-0.5, 6.5)
# Plot the upper levels on ax1 (TURKEY and BUZZY) - make them closer together
ax1.set_ylim(17, 37)

# Hide spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Draw the break marks
d = .015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


# Draw levels and labels with smaller font, positioned above the lines
for name, energy in levels.items():
    # TURKEY and BUZZY go in ax1 (upper), others in ax2 (lower)
    ax = ax1 if name in ['TURKEY', 'BUZZY'] else ax2
    ax.axhline(y=energy, color='k', linestyle='-', linewidth=1.0)
    ax.text(0.98, energy, name, va='bottom', ha='left', fontsize=9)

ax2.set_ylabel(r'Potential $\beta V$', fontsize=10)
# Set y-ticks for upper subplot (TURKEY and BUZZY both shown as ~infinity)
ax1.set_yticks([levels['TURKEY'], levels['BUZZY']])
ax1.set_yticklabels([r'$\sim\infty$', r'$\sim\infty$'], fontsize=10)
# Set y-ticks for lower subplot (ATTITUDE, PERSONAL, PROBLEMS)
ax2.set_yticks([levels['ATTITUDE'], levels['PERSONAL'], levels['PROBLEMS']])
ax2.set_yticklabels([str(levels['ATTITUDE']), str(levels['PERSONAL']), str(levels['PROBLEMS'])], fontsize=10)


# --- Draw arrows for transitions ---
from matplotlib.patches import FancyArrowPatch

# Define the specific order for transitions
ordered_transitions = [
    ('PERSONAL', 'ATTITUDE'),
    ('ATTITUDE', 'PERSONAL'),
    ('PROBLEMS', 'ATTITUDE'),
    ('ATTITUDE', 'PROBLEMS'),
    ('TURKEY', 'ATTITUDE'),
    ('BUZZY', 'ATTITUDE'),
    ('BUZZY', 'TURKEY'),
]

# Assign x-positions for the ordered transitions
x_positions = np.linspace(0.15, 0.85, len(ordered_transitions))

for i, (f, g) in enumerate(ordered_transitions):
    prob = transitions_prob.get((f, g))
    if prob is None:
        continue

    y_f = levels[f]
    y_g = levels[g]
    
    arrow_width = 0.5 + 2 * prob
    x = x_positions[i]
    
    dy = y_g - y_f
    
    # Determine which subplot each level belongs to
    # Upper subplot (ax1): TURKEY (18.2), BUZZY (34.0)
    # Lower subplot (ax2): ATTITUDE (0.0), PERSONAL (4.1), PROBLEMS (5.2)
    f_in_upper = f in ['TURKEY', 'BUZZY']
    g_in_upper = g in ['TURKEY', 'BUZZY']
    
    # Adjust start and end points for arrows
    if f_in_upper and g_in_upper:
        # Both in upper subplot
        start_ax = ax1
        end_ax = ax1
        y_start_arrow = (y_f + 0.3) if dy > 0 else (y_f - 0.3)
        y_end_arrow = (y_g - 0.3) if dy > 0 else (y_g + 0.3)
    elif not f_in_upper and not g_in_upper:
        # Both in lower subplot
        start_ax = ax2
        end_ax = ax2
        y_start_arrow = (y_f + 0.1) if dy > 0 else (y_f - 0.1)
        y_end_arrow = (y_g - 0.1) if dy > 0 else (y_g + 0.1)
    elif f_in_upper and not g_in_upper:
        # From upper to lower (cross-axis)
        start_ax = ax1
        end_ax = ax2
        y_start_arrow = y_f - 0.3
        y_end_arrow = y_g + 0.1
    else:
        # From lower to upper (cross-axis)
        start_ax = ax2
        end_ax = ax1
        y_start_arrow = y_f + 0.1
        y_end_arrow = y_g - 0.3

    # Draw arrows
    if start_ax != end_ax:
        # Cross-axis transitions
        if f_in_upper and not g_in_upper:
            # From upper to lower
            # Draw line from start down to bottom of ax1
            start_ax.plot([x, x], [y_start_arrow, ax1.get_ylim()[0]], 
                         color='k', lw=arrow_width, clip_on=False, zorder=1)
            # Draw line from top of ax2 down to target with arrow
            end_ax.annotate("", xy=(x, y_end_arrow), xytext=(x, ax2.get_ylim()[1]),
                          arrowprops=dict(arrowstyle="->", color="k", lw=arrow_width),
                          clip_on=False, zorder=1)
        else:
            # From lower to upper
            # Draw line from start up to top of ax2
            start_ax.plot([x, x], [y_start_arrow, ax2.get_ylim()[1]], 
                         color='k', lw=arrow_width, clip_on=False, zorder=1)
            # Draw line from bottom of ax1 up to target with arrow
            end_ax.annotate("", xy=(x, y_end_arrow), xytext=(x, ax1.get_ylim()[0]),
                          arrowprops=dict(arrowstyle="->", color="k", lw=arrow_width),
                          clip_on=False, zorder=1)
    else:
        # Transitions within the same subplot
        start_ax.annotate("", xy=(x, y_end_arrow), xytext=(x, y_start_arrow),
                    arrowprops=dict(arrowstyle="->", color="k", lw=arrow_width))

    # Add label for T value - place in the MIDDLE of the arrow
    if start_ax != end_ax:
        # For cross-axis transitions
        if f_in_upper and not g_in_upper:
            # Upper to lower - place label in middle of ax2 portion
            ax_for_label = ax2
            label_y = (ax2.get_ylim()[1] + y_end_arrow) / 2
        else:
            # Lower to upper - place label in middle of ax2 portion
            ax_for_label = ax2
            label_y = (y_start_arrow + ax2.get_ylim()[1]) / 2
    else:
        # Within same region
        ax_for_label = start_ax
        label_y = (y_start_arrow + y_end_arrow) / 2

    # Larger font size with semi-transparent background
    ax_for_label.text(x, label_y, f'{prob:.2f}', ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.0))


ax1.set_xlim(0, 1.2)
ax2.set_xlim(0, 1.2)
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

plt.tight_layout(pad=0.2)
import os
os.makedirs('./figures/claude', exist_ok=True)
plt.savefig("./figures/claude/transition_diagram.pdf")

print("Transition diagram saved as transition_diagram.pdf")