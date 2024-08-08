import pathlib
import shutil

import numpy as np
import pytest
import ase.io
import typing
import random

import ipsuite as ips
from ips_nequip import Nequip

# tests/integration_test/models/test_i_nequip.py == __file__
TEST_PATH = pathlib.Path(__file__).parent.resolve()

@pytest.fixture()
def traj_file(tmp_path_factory, atoms_list) -> str:
    """Save an extxyz trajectory file based on atoms_list."""
    temporary_path = tmp_path_factory.getbasetemp()
    file = temporary_path / "trajectory.extxyz"
    ase.io.write(file, atoms_list)

    return file.as_posix()

@pytest.fixture
def atoms_list() -> typing.List[ase.Atoms]:
    """Generate ase.Atoms objects.

    Construct Atoms objects with random positions and increasing energy
    and random force values.
    """
    random.seed(1234)
    atoms = [
        ase.Atoms(
            "CO",
            positions=[(0, 0, 0), (0, 0, random.random())],
            cell=(1, 1, 1),
            pbc=True,
        )
        for _ in range(21)
    ]

    for idx, atom in enumerate(atoms):
        atom.calc = ase.calculators.singlepoint.SinglePointCalculator(
            atoms=atom,
            energy=idx / 21,
            forces=np.random.randn(2, 3),
            energy_uncertainty=idx + 2,
            forces_uncertainty=np.full((2, 3), 2.0) + idx,
        )

    return atoms


def test_model_training(proj_path, traj_file):
    shutil.copy(TEST_PATH / "allegro_minimal.yaml", proj_path / "allegro_minimal.yaml")
    with ips.Project() as project:
        data_1 = ips.nodes.AddData(file=traj_file, name="data_1")

        train_selection = ips.nodes.UniformEnergeticSelection(
            data=data_1.atoms, n_configurations=10, name="train_data"
        )

        validation_selection = ips.nodes.UniformEnergeticSelection(
            data=train_selection.excluded_atoms, n_configurations=8, name="val_data"
        )

        model = Nequip(
            config="allegro_minimal.yaml",
            device="cpu",
            data=train_selection.atoms,
            validation_data=validation_selection.atoms,
        )

    project.run()

    data_1.load()

    model.load()

    atoms = data_1.atoms[0]
    atoms.calc = model.get_calculator()

    assert isinstance(atoms.get_potential_energy(), float)
    assert atoms.get_potential_energy() != 0.0

    assert isinstance(atoms.get_forces(), np.ndarray)
    assert atoms.get_forces()[0, 0] != 0.0

    assert model.lammps_pair_style == "allegro"
    assert model.lammps_pair_coeff[0] == "* * nodes/MLModel/deployed_model.pth C O"
