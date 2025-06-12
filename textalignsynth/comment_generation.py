import numpy as np


def get_measure_comments(measure_annot_list, start=1, step=1):
    """
    This method converts a measure annotation list into a measure comment list.
    Given measure numbers are converted into strings and the measure annotations are reduced
    according to the given start/step values.

    Parameters:
        measure_annot_list      list of the form [[float: time in seconds, float: measure number], ...],
                                    e.g., [[0.4, 1], [1.3, 2]]
        start                   number of the first measure to be synthesized
        step                    interval between syntesized measure numbers
    """
    measure_comment_list = []

    for i in range(len(measure_annot_list)):
        t_start, i_measure = measure_annot_list[i]

        # check that measure number is integer and leave out some measures if necessary
        if i_measure.is_integer() and (int(i_measure - start) % step == 0):
            measure_comment_list.append([t_start, f"{int(i_measure)}."])

    return measure_comment_list


def get_chord_comments(chord_annot_list, filter_valid=True, remove_repeated=True):
    """
    This method converts a chord annotation list into a chord comment list. Chord annotations
    are converted into comments by using the stored dictionary 'self.chord_synth_dict'.
    Optionally, invalid chords (for which there is no key in the dictionary) and chord repetitions are removed.

    Parameters:
        chord_annot_list        list of the form [[float: time in seconds, str: chord annotation], ...],
                                    e.g., [[0.4, 'G'], [1.3, 'Bb:min']]
        filter_valid            whether to remove invalid chords
        remove_repeated         whether to remove consecutive chord repetitions
    """

    # ----------------------  Initialize chord dict ---------------------- #
    chroma_base = ["C", "D", "E", "F", "G", "A", "B"]

    # add sharp and flat
    chroma_labels = chroma_base + [f"{c}#" for c in chroma_base] + [f"{c}b" for c in chroma_base]

    # extension for major / minor
    chord_labels = (
        chroma_labels + [f"{c}:maj" for c in chroma_labels] + [f"{c}:min" for c in chroma_labels]
    )

    # write everything out
    chord_symb_dict = {
        f"{c}": c.replace("#", " sharp").replace("b", " flat") for c in chord_labels
    }
    chord_symb_dict = {
        f"{k}": (
            v.replace(":maj", " major.").replace(":min", " minor.")
            if ":maj" in v or ":min" in v
            else f"{v} major."
        )
        for k, v in chord_symb_dict.items()
    }
    chord_symb_dict = {f"{k}": f"{v[0]}.{v[1:]}" for k, v in chord_symb_dict.items()}

    # ----------------------  Convert chord annotation list ---------------------- #
    # only keep chords for which a key exists in the chord dict
    if filter_valid:
        chord_annot_list = [x for x in chord_annot_list if x[1] in chord_symb_dict.keys()]

    # remove repetitions
    if remove_repeated:
        chord_annot_np = np.array(chord_annot_list, dtype=object)

        # Get boolean mask for rows where the string is not equal to the previous row
        mask = np.append(True, chord_annot_np[1:, 1] != chord_annot_np[:-1, 1])

        # Use mask to filter out rows
        chord_annot_list = chord_annot_np[mask].tolist()

    chord_comment_list = [
        [t, chord_symb_dict[c] if c in chord_symb_dict.keys() else c] for t, c in chord_annot_list
    ]

    return chord_comment_list


def get_leitmotif_comments(leitmotif_annot_list):
    """
    This method converts a leitmotif annotation list into a leitmotif comment list.
    The method only appends a '.' to the leitmotif name for enhanced prosody.

    Parameters:
        measure_annot_list      list of the form [[float: time in seconds, string: leitmotif name], ...],
                                    e.g., [[43.581995464, 'Ring'], [245.39, 'Horn']]
    """
    leitmotif_comment_list = []

    for t_start, motif in leitmotif_annot_list:
        leitmotif_comment_list.append([t_start, f"{motif}."])

    return leitmotif_comment_list


def get_structure_comments(structure_annot_list):
    """
    This method converts a structure annotation list into a structure comment list.
    The method converts colons into spaces and appends a '.' for enhanced prosody.

    Parameters:
        structure_annot_list      list of the form [[float: time in seconds, string: structural annotation], ...],
                                    e.g., [[43.581995464, 'Exposition: First Group'], [245.39, 'Horn.']]
    """
    structure_comment_list = []

    for t_start, struct in structure_annot_list:
        structure_comment_list.append([t_start, f'{struct.replace(":", " ")}.'])

    return structure_comment_list
