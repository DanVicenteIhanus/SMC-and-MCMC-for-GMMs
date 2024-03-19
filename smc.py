import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import seaborn as sns
import pandas as pd

sns.set_theme(style="white", rc={"axes.facecolor":(0,0,0,0)})
PATH = "FILEPATH"

def gen_obs(T, n_particles) -> np.ndarray:    
    x = np.zeros((T, n_particles))
    y = np.zeros((T, n_particles))
    x[0] = np.random.normal(loc=0, scale=np.sqrt(10), size=n_particles)
    y[0] = (x[0]**2)/20 + np.random.normal(loc=0, scale=1, size=n_particles)
    for t in range(T - 1):
        w_t = np.random.normal(loc=0, scale=1)
        x[t + 1, :] = gen_x(x[t], t, n_particles)
        y[t + 1, :] = (1/20) * x[t + 1]**2 + w_t
    return x, y

def gen_x(x, t, n_particles) -> float:
    v_t = np.random.normal(loc=0, scale=np.sqrt(10), size=n_particles)
    x_new = 0.5 * x  + 25 * (x / (1 + x**2)) + 8*np.cos(1.2*t) + v_t
    return x_new

def y_obs(x, n_particles) -> float:
    w = np.random.normal(loc=0, scale=1, size=n_particles)
    y = (x**2)/20 + w
    return y

def bootstrap_filter(n_particles, T, gen_x, y_obs) -> np.ndarray:
    """args: 
    n_particles : number of particles in filter
    T           : end-time
    gen_x       : dynamics
    y_obs       : observation model
    returns:    : x (samples from distribution)
    """
    
    # INIT:
    #w       = np.zeros((T, n_particles))
    x       = np.zeros((T, n_particles))
    _, y    = gen_obs(T, n_particles)
    x[0, :] = np.random.normal(loc=0, scale=np.sqrt(10), size=n_particles)
    y[0, :] = y_obs(x[0, :], n_particles)
    w = np.zeros((T, n_particles))
    w[0, :] = 1.0 / n_particles  # Initialize weights

    for t in tqdm(range(1, T)):
        # == Importance Sampling step == #
        x[t] = gen_x(x[t - 1], t, n_particles)      # Vectorized state update
        y_pred = (x[t]**2) / 20                     # Vectorized observation prediction
        w[t] = norm.pdf(y[t], loc=y_pred, scale=1)  # Vectorized weight calculation
        w[t] /= np.sum(w[t])                        # Normalize weights

        # == Resampling step == #
        indices = np.random.choice(n_particles, n_particles, p=w[t])
        x[t, :] = x[t, indices]                     # resample n_particles particles {x_0:t^{1}, ..., n_particles}
    return x, w

if __name__ == "__main__":
    T = 200
    t = np.arange(0, T, 1)

    x_0 = np.random.normal(loc=0, scale = np.sqrt(10)) # gen init val
    w_t = np.random.normal(loc=0, scale=1)
    y_0 = x_0**2/20 + w_t
    
    x, y = gen_obs(T, 1)
    
    # Creating the trajectory plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(t, y, lw=0.5, color='k')
    ax.set_xlabel('$t$')
    ax.set_xlim([0, T])
    ax.set_ylabel('$y(t)$')
    ax.set_title('Timeseries $y(t)$')

    plt.show()
    
    n_particles = 50000
    x, _ = bootstrap_filter(n_particles, T, gen_x, y_obs)
    selected_time_steps = [20*i for i in range(9)] 
    selected_time_steps.append(199)
    
    # Create a DataFrame for plot
    df_list = []
    for t in selected_time_steps:
        for i in range(n_particles):
            df_list.append({'time': f'time {t}', 'x(t)': x[t, i]})
    df = pd.DataFrame(df_list)
    palette = sns.color_palette("Blues", len(selected_time_steps))
    g = sns.FacetGrid(df, row="time", hue="time", aspect=15, height=0.5, palette=palette)
    g.map(sns.kdeplot, "x(t)", bw_adjust=.5, clip_on=False, fill=True) # bw_adjust=.5 <- add this for sns.kdeplot bins=100, kde=True, <- add this for sns.histplot
    
    # Set y-axis to show timesteps
    for i, ax in enumerate(g.axes.flat):
        ax.set_ylabel(f"t = {selected_time_steps[i]}")
        label = ax.get_yaxis().get_label()
        label.set_rotation(0) 
        label.set_horizontalalignment('right')
        label.set_position((0, 0.2)) 

    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    g.savefig(PATH + f"plots/filtering_density_N_=_{n_particles}.pdf")
    
    """
    plt.show()
    for t in selected_time_steps:
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x[t, :], bw_adjust=0.5, fill=True)  # Adjust bw_adjust for bandwidth
        plt.xlim(-25, 25)
        plt.title(f'Estimated Filtering Distribution at Time {t}')
        plt.xlabel('$x_{t}$')
        plt.ylabel('Density')
        plt.show()
    """
