# coding: utf-8
# Copyright (c) SJTU Pymatgen Development Team.

from __future__ import division, unicode_literals, print_function

import logging
from collections import defaultdict

import numpy as np
from monty.io import zopen

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element

from monty.json import MSONable

"""
Classes for reading/manipulating/writing lammps atom-style dump files.
"""

__author__ = "Lingti Kong"
__credits__ = "Lingti Kong"
__copyright__ = "Copyright 2016, SJTU MaGIC"
__version__ = "1.0"
__maintainer__ = "Lingti Kong"
__email__ = "konglt@gmail.com"
__status__ = "Development"
__date__ = "Mar 1, 2016"

logger = logging.getLogger(__name__)

class lmpdump(MSONable):
    """
    Vastly improved cElementTree-based parser for vasprun.xml files. Uses
    iterparse to support incremental parsing of large files.
    Speedup over Dom is at least 2x for smallish files (~1Mb) to orders of
    magnitude for larger files (~10Mb).

    Args:
        filename (str): Filename to parse
        ionic_step_skip (int): If ionic_step_skip is a number > 1,
            only every ionic_step_skip ionic steps will be read for
            structure and energies. This is very useful if you are parsing
            very large vasprun.xml files and you are not interested in every
            single ionic step. Note that the final energies may not be the
            actual final energy in the vasprun.
        ionic_step_offset (int): Used together with ionic_step_skip. If set,
            the first ionic step read will be offset by the amount of
            ionic_step_offset. For example, if you want to start reading
            every 10th structure but only from the 3rd structure onwards,
            set ionic_step_skip to 10 and ionic_step_offset to 3. Main use
            case is when doing statistical structure analysis with
            extremely long time scale multiple VASP calculations of
            varying numbers of steps.
         symbols (list of string)

    **lmp dump results**

    .. attribute:: ionic_steps

        All ionic steps in the run as a list of
        {"structure": structure at end of run,
        "md_step": MD step number}

    .. attribute:: structures

        List of Structure objects for the structure at each ionic step.

    Author: Lingti Kong
    """

    def __init__(self, structures, timesteps=None):
        if not timesteps:
            n = len(structures)
            timesteps = [i for i in range(n+1)]
        ionic_steps = []
        for i,structure in enumerate(structures):
            ionic_steps.append({"structure":structure, "timestep":timesteps[i]})

        self.ionic_steps = ionic_steps
        
    def get_string(self, significant_figures=10):
        """
        Returns a string to be written as a lammps atom-style dump file.

        Args:
            significant_figures (int): No. of significant figures to
                output all quantities. Defaults to 10.

        Returns:
            String representation of the atom-style dump file.
        """

        types_of_specie = []
        for ionic_step in self.ionic_steps:
            structure = ionic_step.get("structure")
            for itype in structure.types_of_specie:
                if itype not in types_of_specie:
                    types_of_specie.append(itype)

        lines = []
        for ionic_step in self.ionic_steps:
            structure = ionic_step.get("structure")
            timestep  = ionic_step.get("timestep")
            lines.append("ITEM: TIMESTEP")
            lines.append(str(timestep))
            lines.append("ITEM: NUMBER OF ATOMS")
            lines.append(str(structure.num_sites))
            ((a, b, c), (alpha, beta, gamma)) = structure.lattice.lengths_and_angles
            lx = a
            xy = b * np.cos(gamma*np.pi/180)
            xz = c * np.cos(beta*np.pi/180)
            ly = np.sqrt(b*b - xy*xy)
            yz = (b*c*np.cos(alpha*np.pi/180) - xy*xz)/ly
            lz = np.sqrt(c*c - xz*xz - yz*yz)
            triclinic = (xy*xy + xz*xz + yz*yz) > 1.e-15
            format_str = "{{:.{0}f}}".format(significant_figures)
            if triclinic:
               lines.append("ITEM: BOX BOUNDS xx yy zz xy xz yz")
               lines.append(" ".join([format_str.format(c) for c in (0, lx, xy)]))
               lines.append(" ".join([format_str.format(c) for c in (0, ly, xz)]))
               lines.append(" ".join([format_str.format(c) for c in (0, lz, yz)]))
            else:
               lines.append("ITEM: BOX BOUNDS xx yy zz")
               lines.append(" ".join([format_str.format(c) for c in (0, lx)]))
               lines.append(" ".join([format_str.format(c) for c in (0, ly)]))
               lines.append(" ".join([format_str.format(c) for c in (0, lz)]))
            lines.append("ITEM: ATOMS id type xs ys zs")

            for (i, site) in enumerate(structure):
                coords = site.frac_coords
                specie = site.specie
                itype  = types_of_specie.index(specie) + 1
                line = " ".join([str(i+1), str(itype)]) + " "
                line += " ".join([format_str.format(c) for c in coords])
                lines.append(line)

        return "\n".join(lines) + "\n"

    def __repr__(self):
        return self.get_string()

    def __str__(self):
        """
        String representation of lammp atom style dump file.
        """
        return self.get_string()

    def write_file(self, filename, **kwargs):
        """
        Writes lammps atom style dump info to a file
        """
        with zopen(filename, "wt") as f:
            f.write(self.get_string(**kwargs))

    @property
    def structures(self):
        return [step["structure"] for step in self.ionic_steps]

    @property
    def final_structure(self):
        return self.ionic_steps[-1]["structure"]

    @property
    def nimages(self):
        return len(self.ionic_steps)

    @property
    def lattice(self):
        return self.final_structure.lattice

    @property
    def lattice_rec(self):
        return self.final_structure.lattice.reciprocal_lattice

    def write_file(self, filename):
        """
        Writes lammps atom style dump to a file.
        """
        with zopen(filename, "wt") as f:
            f.write(self.get_string())

    def as_dict(self):
        """
        Json-serializable dict representation.
        """
def from_file(filename, ionic_step_skip=None,
              ionic_step_offset=0, symbols=None):
    """
    Read a lammps atom style dump file.
    """
    with zopen(filename, "rt") as f:
        if ionic_step_skip or ionic_step_offset:
            # remove parts of the dump file and parse the string
            run = f.read()
            steps = run.split("ITEM: TIMESTEP")
            #Nothing before the first <ITEM:TIMESTEP>
            steps.pop(0)
            new_steps = steps[ionic_step_offset::int(ionic_step_skip)]
            #add the tailing informat in the last step from the run
            to_parse = "ITEM: TIMESTEP".join(new_steps)
            if steps[-1] != new_steps[-1]:
                to_parse = "ITEM: TIMESTEP\n{}{}".format( to_parse, steps[-1])
            else:
                to_parse = "ITEM: TIMESTEP\n{}".format(to_parse)
            dump = from_string(to_parse, symbols)
        else:
            dump = from_string(f.read(), symbols)
    return dump

def from_structures(structures):
    return lmpdump.lmpdump(structures)

def from_string(string, symbols=None):
    structures = []
    timesteps = []
    steps = string.split("ITEM: TIMESTEP")
    steps.pop(0)
    for step in steps:
        lines = tuple(step.split("\n"))
        mdstep = int(lines[1])
        natoms = int(lines[3])
        xbox   = tuple((lines[5] + " 0").split())
        ybox   = tuple((lines[6] + " 0").split())
        zbox   = tuple((lines[7] + " 0").split())
        xlo = float(xbox[0])
        xhi = float(xbox[1])
        xy  = float(xbox[2])
        ylo = float(ybox[0])
        yhi = float(ybox[1])
        xz  = float(ybox[2])
        zlo = float(zbox[0])
        zhi = float(zbox[1])
        yz  = float(zbox[2])
        xlo -= np.min([0, xy, xz, xy+xz])
        xhi -= np.max([0, xy, xz, xy+xz])
        ylo -= np.min([0, yz])
        yhi -= np.max([0, yz])
      
        lattice = [xhi-xlo, 0, 0, xy, yhi-ylo, 0, xz, yz, zhi-zlo]
        atoms = [[float(j) for j in line.split()] for line in lines[9:-1]]
        atoms = sorted(atoms, key=lambda a_entry: a_entry[0], reverse=True) 

        coords = []
        atomic_sym = []
        while (len(atoms)):
           one = atoms.pop()
           typ = one[1]
           coo = one[2:5]
           sym = symbols[typ] if symbols else Element.from_Z(typ).symbol
           atomic_sym.append(sym)
           coords.append(coo)
        struct = Structure(lattice, atomic_sym, coords,
                   to_unit_cell=False, validate_proximity=False,
                   coords_are_cartesian=False)
        structures.append(struct)
        timesteps.append(mdstep)
        
    return lmpdump(structures, timesteps)

if __name__ == "__main__":
   x = from_file("dump.lammpstrj")
   # print(x)
   print(x.lattice)
   print(x.nimages)
   x.write_file("new.dump")
