import numpy as np
import matplotlib.pyplot as plt

# Drug properties (from data in paper 3)
drug = {
    "CP": {  # Cyclophosphamide
        "Sensitivity": 0.63,  # [nA/µM*mm^2]
        "Peak_pos": -296,  # [mV]
        "Peak_width": 400,  # [mV]  OG = 100
        "k_m": 7000,  # [µM]
        "v_max": 2.3,  # [nmol/min/mg]
        "full_name": "Cyclophosphamide",
    },
    "BUSU": {  # Busulfan
        "Sensitivity": 0,  # [nA/µM*mm^2]
        "Peak_pos": 0,  # [mV]
        "Peak_width": 0,  # [mV]
        "k_m": 0,  # [µM]
        "v_max": 0,  # [nmol/min/mg]
        "full_name": "Busulfan",
    },
    "MT": {  # Methotrexate
        "Sensitivity": 0,  # [nA/µM*mm^2]
        "Peak_pos": 0,  # [mV]
        "Peak_width": 0,  # [mV]
        "k_m": 0,  # [µM]
        "v_max": 0,  # [nmol/min/mg]
        "full_name": "Methotrexate",
    },
    "5FLUO": {  # 5-Fluorouracil
        "Sensitivity": 0,  # [nA/µM*mm^2]
        "Peak_pos": 0,  # [mV]
        "Peak_width": 0,  # [mV]
        "k_m": 0,  # [µM]
        "v_max": 0,  # [nmol/min/mg]
        "full_name": "5-Fluorouracil",
    },
    "IFOS": {  # Ifosfamide
        "Sensitivity": 0.4,  # [nA/µM*mm^2]
        "Peak_pos": -450,  # [mV]
        "Peak_width": 150,  # [mV]
        "k_m": 8.1,  # [µM]
        "v_max": 1.2,  # [nmol/min/mg]
        "full_name": "Ifosfamide",
    },
    "ETOP": {  # Etoposide
        "Sensitivity": 9.1,  # [nA/µM*mm^2]
        "Peak_pos": 220,  # [mV]
        "Peak_width": 150,  # [mV]
        "k_m": 77.7,  # [µM]
        "v_max": 314,  # [nmol/min/mg]
        "full_name": "Etoposide",
    },
    "FLUR": {  # flurbiprofen
        "Sensitivity": 0.46,  # [nA/µM*mm^2]
        "Peak_pos": 60,  # [mV]
        "Peak_width": 200,  # [mV] # OG = 200
        "k_m": 1.9,  # [µM]
        "v_max": 343,  # [nmol/min/mg]
        "full_name": "Flurbiprofen",
    },
    "NAP": {  # naproxen
        "Sensitivity": 0.25,  # [nA/µM*mm^2]
        "Peak_pos": -10,  # [mV]
        "Peak_width": 200,  # [mV]
        "k_m": 143,  # [µM]
        "v_max": 0.84,  # [nmol/min/mg]
        "full_name": "Naproxen",
    },
    "DX": {  # naproxen
        "Sensitivity": 6.63,  # [nA/µM*mm^2]
        "Peak_pos": -392,  # [mV]
        "Peak_width": 250,  # [mV]  OG = 200
        "k_m": 1155,  # [µM]
        "v_max": 11.9,  # [nmol/min/mg]
        "full_name": "Dextromethorphan",
    },
}

################


def activation(
    drug1,
    drug2,
    Cs,
    peak_amplitude=[10, 10],
    interference_strength=0.5,
    width_adjustment=4,
    potential_range=100, 
    adaptative_potential=False,
    THRESHOLD_POTENTIAL=150
):
    """
    This function calculates the current I as a function of the potential V for two drugs.

    Parameters:
    drug1 (str): Name of the first drug
    drug2 (str): Name of the second drug
    Cs (list): List of concentrations for the two drugs in µM
    peak_amplitude (list, optional): List of peak amplitudes A_k for the two drugs in nA/µM*mm^2. Defaults to [10, 10].
    interference_strength (float, optional): Coefficient for interference between the two drugs. Defaults to 0.5.
    width_adjustment (int, optional): Adjustment factor for peak width. Defaults to 4.
    potential_range (int, optional): Range of potential values in mV. Defaults to 100.

    Returns:
    tuple: Current and potential arrays
    """

    A = 12.56  # Electrode area in mm^2

    # Define potential range
    potential = np.linspace(-600, 600, 1200)

    if adaptative_potential==True:
        min_peak_pos = min(drug[drug1]["Peak_pos"], drug[drug2]["Peak_pos"])
        max_peak_pos = max(drug[drug1]["Peak_pos"], drug[drug2]["Peak_pos"])
        potential = np.linspace(min_peak_pos-THRESHOLD_POTENTIAL, max_peak_pos+THRESHOLD_POTENTIAL, abs(min_peak_pos-THRESHOLD_POTENTIAL)+abs(max_peak_pos+THRESHOLD_POTENTIAL) )


    # Sensitivity of the two drugs
    sensitivity = [drug[drug1]["Sensitivity"], drug[drug2]["Sensitivity"]]

    # Peak position for the two drugs
    peakposition = [drug[drug1]["Peak_pos"], drug[drug2]["Peak_pos"]]
    # Peak width for the two drugs
    peakwidth = [drug[drug1]["Peak_width"], drug[drug2]["Peak_width"]]

    # Calculate the rate of reaction for the first drug
    rate = Cs[0] / (drug[drug1]["k_m"] + Cs[0])

    # Compute the adjusted amplitude considering interference
    ampli = peak_amplitude[1] * (1 + rate * interference_strength)

    # Lists to store the current values
    I1 = []
    I2 = []

    # Compute the current for each potential value
    for i, v in enumerate(potential):
        I1.append(
            0.4463
            * sensitivity[0]
            * A
            * Cs[0]
            * peak_amplitude[0]
            * np.exp(
                -((v - peakposition[0]) ** 2) / ((peakwidth[0] / width_adjustment) ** 2)
            )
        )
        I2.append(
            0.4463
            * sensitivity[1]
            * A
            * Cs[1]
            * ampli
            * np.exp(
                -((v - peakposition[1]) ** 2) / ((peakwidth[1] / width_adjustment) ** 2)
            )
        )

    # Return the sum of the currents and the potential
    return np.array(I1) + np.array(I2), potential


def inhibition(
    drug1,
    drug2,
    Cs,
    peak_amplitude=[10, 10],
    interference_strength=0.5,
    width_adjustment=4,
    potential_range=100,
    adaptative_potential=False,
    THRESHOLD_POTENTIAL=150
):
    """
    This function calculates the current I as a function of the potential V for two drugs.

    Parameters:
    ----------------
    drug1 (str): Name of the first drug
    drug2 (str): Name of the second drug
    Cs (list): List of concentrations for the two drugs in µM
    peak_amplitude (list, optional): List of peak amplitudes A_k for the two drugs in nA/µM*mm^2. Defaults to [10, 10].
    interference_strength (float, optional): Coefficient for interference between the two drugs. Defaults to 0.5.
    width_adjustment (int, optional): Adjustment factor for peak width. Defaults to 4.
    potential_range (int, optional): Range of potential values in mV. Defaults to 100.

    Returns:
    ----------------
    tuple: Current and potential arrays
    """

    A = 12.56  # Electrode area in mm^2

    # Define potential range
    potential = np.linspace(-600, 600, 1200)

    if adaptative_potential==True:
        min_peak_pos = min(drug[drug1]["Peak_pos"], drug[drug2]["Peak_pos"])
        max_peak_pos = max(drug[drug1]["Peak_pos"], drug[drug2]["Peak_pos"])
        potential = np.linspace(min_peak_pos-THRESHOLD_POTENTIAL, max_peak_pos+THRESHOLD_POTENTIAL, abs(min_peak_pos-THRESHOLD_POTENTIAL)+abs(max_peak_pos+THRESHOLD_POTENTIAL) )

    # Sensitivity of the two drugs
    sensitivity = [drug[drug1]["Sensitivity"], drug[drug2]["Sensitivity"]]

    # Peak position for the two drugs
    peakposition = [drug[drug1]["Peak_pos"], drug[drug2]["Peak_pos"]]
    # Peak width for the two drugs
    peakwidth = [drug[drug1]["Peak_width"], drug[drug2]["Peak_width"]]

    # Calculate the rate of reaction for the first drug
    rate = Cs[0] / (drug[drug1]["k_m"] + Cs[0])

    # Compute the adjusted amplitude considering interference
    ampli = peak_amplitude[1] * (1 - rate * interference_strength)

    # Lists to store the current values
    I1 = []
    I2 = []

    # Compute the current for each potential value
    for i, v in enumerate(potential):
        I1.append(
            0.4463
            * sensitivity[0]
            * A
            * Cs[0]
            * peak_amplitude[0]
            * np.exp(
                -((v - peakposition[0]) ** 2) / ((peakwidth[0] / width_adjustment) ** 2)
            )
        )
        I2.append(
            0.4463
            * sensitivity[1]
            * A
            * Cs[1]
            * ampli
            * np.exp(
                -((v - peakposition[1]) ** 2) / ((peakwidth[1] / width_adjustment) ** 2)
            )
        )

    # Return the sum of the currents and the potential
    return np.array(I1) + np.array(I2), potential

# Define a function to compute current for different interactions
def compute_interaction(drug1, drug2, concentrations, interaction_func, fixed_concentration, peak_amplitude=[1.6, 1.45], adaptative_potential=False, THRESHOLD_POTENTIAL=250):
    """
    Computes the current for a list of concentrations based on the specified interaction function.

    Parameters:
    drug1 (str): Name of the first drug
    drug2 (str): Name of the second drug
    concentrations (list): List of concentrations for the first drug
    interaction_func (function): Function to use for computing the interaction (e.g., activation, inhibition)

    Returns:
    list: List of computed currents for each concentration
    """

    _, potential = interaction_func(
        drug1=drug1,
        drug2=drug2,
        Cs=[concentrations[0], fixed_concentration],
        peak_amplitude= peak_amplitude,
        adaptative_potential=adaptative_potential,
        THRESHOLD_POTENTIAL=THRESHOLD_POTENTIAL
    )

    I = [
        interaction_func(
            drug1=drug1,
            drug2=drug2,
            Cs=[concentration, fixed_concentration],
            peak_amplitude= peak_amplitude,
            adaptative_potential=adaptative_potential,
            THRESHOLD_POTENTIAL=THRESHOLD_POTENTIAL
        )[0]
        for concentration in concentrations
    ]

    return I, potential


## LINEAR MODEL ##
def linear(x, slope, intercept):
    return slope * x + intercept


def cyclic_voltammogram(
    potential_range,
    faradaic_current,
    scan_rate=0.001,
    capacitance=4,
    num_cycles=3,
    isp0=0,
    ipc0=0,
):
    """
    This function generates a cyclic voltammogram for the given input parameters.

    Parameters:
    ------------
    potential_range (numpy array): An array representing the range of potential values
    faradaic_current (numpy array): An array representing the faradaic current values
    scan_rate (float): The scan rate
    capacitance (float): The capacitance
    num_cycles (int, optional): Number of cycles to perform. Defaults to 3.
    isp0 (float, optional): Initial value for the forward scan. Defaults to 0.
    ipc0 (float, optional): Initial value for the reverse scan. Defaults to 0.

    Returns:
    ------------
    tuple: potential and current arrays of the cyclic voltammogram
    """

    # Compute the linear capacitance
    slope = capacitance / (potential_range[-1] - potential_range[0])
    intercept = -slope * potential_range[0]

    # Calculate the capacitive current
    capacitive_current = linear(potential_range, slope, intercept) * np.gradient(
        potential_range, scan_rate
    )

    # Forward and reverse faradaic currents
    forward_faradaic_current = faradaic_current
    reverse_faradaic_current = [-x for x in faradaic_current]

    # Combine potential and current data for forward and reverse scans
    forward_scan = np.column_stack(
        (potential_range, forward_faradaic_current + capacitive_current)
    )
    reverse_scan = np.column_stack(
        (potential_range[::-1], reverse_faradaic_current + capacitive_current[::-1])
    )

    # Stack the forward and reverse scans to create the full scan
    full_scan = np.vstack((forward_scan, reverse_scan))
    for _ in range(1, num_cycles):
        full_scan = np.vstack((full_scan, forward_scan, reverse_scan))

    # Return the potential and current data
    return full_scan[:, 0], full_scan[:, 1]


def plot_cyclic_voltammogram(
    drug1,
    drug2,
    potential_range,
    current,
    concentration,
    scan_rate=0.001,
    capacitance=4,
    num_cycles=3,
):
    """
    This function plots a cyclic voltammogram for the given input parameters.

    Parameters:
    ------------
    drug1 (str): Name of the first drug
    drug2 (str): Name of the second drug
    potential_range (numpy array): An array representing the range of potential values
    current (numpy array): An array representing the current values
    concentration (numpy array): An array representing the concentration values
    scan_rate (float, optional): The scan rate. Defaults to 0.001.
    capacitance (float, optional): The capacitance. Defaults to 4.
    num_cycles (int, optional): Number of cycles to perform. Defaults to 3.

    Returns:
    ------------
    tuple: potential and current arrays of the cyclic voltammogram
    """

    # Create a new figure
    plt.figure()

    # Assign current to faradaic current
    faradaic_current = current

    # Generate cyclic voltammogram data
    potential_data, cyclic_current_data = cyclic_voltammogram(
        potential_range,
        faradaic_current,
        scan_rate=scan_rate,
        capacitance=capacitance,
        num_cycles=num_cycles,
    )
    # Plot the cyclic voltammogram
    plt.plot(
        potential_data, cyclic_current_data, label=f"{drug1} {concentration[0]} µM"
    )

    # Print the lengths of the cyclic current and potential data for debugging purposes
    print(len(cyclic_current_data))
    print(len(potential_data))

    # Set the labels and title for the plot
    plt.xlabel("Potential (mV)")
    plt.ylabel("Current (nA)")
    plt.title(
        "Cyclic Voltammogram of {} (variable) and {} (constant)".format(drug[drug1]["full_name"], drug[drug2]["full_name"])
    )

    # Add a legend to the plot
    plt.legend()

    # Add a grid to the plot
    plt.grid()

    # Show the plot
    plt.show()

    # Return the potential and current data
    return potential_data, cyclic_current_data

def plot_interpolated_CV(
    drug1, 
    drug2, 
    concentration, 
    currents, 
    potential_range, 
    potential,
    cyclic_voltammogram_func
):
    """
    This function plots interpolated cyclic voltammograms for the given input parameters.

    Parameters:
    ------------
    drug1 (str): Name of the first drug.
    drug2 (str): Name of the second drug.
    concentration (list): List of concentrations for the first drug.
    currents (list): List of current arrays for each concentration.
    potential_range (numpy array): An array representing the range of potential values.
    potential (numpy array): An array representing the potential values.
    cyclic_voltammogram_func (function): The function to generate cyclic voltammogram data.

    Returns:
    ------------
    None
    """

    # Create a new figure
    plt.figure()

    # Plot each concentration's cyclic voltammogram
    for i, conc in enumerate(concentration):
        faradaic_current = currents[i]

        # Interpolate the faradaic_current values to match the length of potential_range
        faradaic_current_interp = np.interp(potential_range, potential, faradaic_current)

        # Generate cyclic voltammogram data and plot it
        potential_data, cyclic_current_data = cyclic_voltammogram_func(potential_range, faradaic_current_interp)
        plt.plot(potential_data, cyclic_current_data, label=f"{drug1} {conc} µM")

    # Set the labels and title for the plot
    plt.xlabel("Potential (mV)")
    plt.ylabel("Current (nA)")
    plt.title("Cyclic Voltammogram of {} ({}) (variable) \n and {} ({}) (fixed)".format(drug[drug1]["full_name"], drug1, drug[drug2]["full_name"], drug2))

    # Add a legend to the plot
    plt.legend()

    # Add a grid to the plot
    plt.grid()

    # Show the plot
    plt.show()


def plot_acti_inhi(drug1, drug2, concentration, activation, inhibition, potential):

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for i, conc in enumerate(concentration):
        axs[0].plot(potential, activation[i], label="{}: {} µM".format(drug[drug1]["full_name"], conc))
        axs[1].plot(potential, inhibition[i], label="{}: {} µM".format(drug[drug1]["full_name"], conc))

    axs[0].set_xlabel("Potential (mV)")
    axs[1].set_xlabel("Potential (mV)")
    axs[0].set_ylabel("Current (nA)")
    axs[0].set_title("Current vs potential (activation) for \n {} ({}) fixed, {} ({}) variable".format(drug[drug2]["full_name"], drug2, drug[drug1]["full_name"], drug1))
    axs[1].set_title("Current vs potential (inhibition) for \n {} ({}) fixed, {} ({}) variable".format(drug[drug2]["full_name"], drug2, drug[drug1]["full_name"], drug1))

    # add a vertical line at the peak position
    for ax in axs:
        ax.axvline(x=drug[drug1]["Peak_pos"], color='k', linestyle='--')
        ax.axvline(x=drug[drug2]["Peak_pos"], color='k', linestyle='--')
        ax.text(drug[drug1]["Peak_pos"]+10, 0, "{} peak position".format(drug1), rotation=90)
        ax.text(drug[drug2]["Peak_pos"]+10, 0, "{} peak position".format(drug2), rotation=90)

    axs[0].legend()
    axs[1].legend()
    # change x axis limits
    #axs[0].set_xlim([-150, 250])
    #axs[1].set_xlim([-150, 250])
    plt.show()