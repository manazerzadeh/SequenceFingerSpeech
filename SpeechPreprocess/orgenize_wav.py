    
from pathlib import Path
import shutil
import re


def move_eff0_files(base_dir, SN_list, BN_list, Eff_list):
    """
    Move wav files with Eff==0 into base_dir/SNx/BNy folders.

    Parameters
    ----------
    base_dir : str or Path
        Directory containing the wav files.
    SN_list : list[int]
        List of SN numbers to include.
    BN_list : list[int]
        List of BN numbers to include.
    """

    base_dir = Path(base_dir)
    pattern = re.compile(r"SN(\d+)_BN(\d+)_TN\d+_Eff(\d+)\.wav")

    for wav_file in base_dir.glob("*.wav"):

        match = pattern.match(wav_file.name)
        if not match:
            continue

        SN = int(match.group(1))
        BN = int(match.group(2))
        Eff = int(match.group(3))

        if SN in SN_list and BN in BN_list and Eff in Eff_list:
            target_dir = base_dir / f"SN{SN}" / f"BN{BN}"
            target_dir.mkdir(parents=True, exist_ok=True)

            shutil.move(wav_file, target_dir / wav_file.name)
            
if __name__ == "__main__":
    
    base_dir = r"Y:\data\SequenceFingerSpeech\raw\SFS1\speech"

    SN_list = [6,7,8,9,10,12,13,14,15,16,17]
    BN_list = [2,3,4,5,6,7,8,9,10,11,12,13]
    Eff_list = [0]

    move_eff0_files(base_dir, SN_list, BN_list, Eff_list)