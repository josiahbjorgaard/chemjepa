import numpy as np
from rdkit import RDLogger, Chem
from utils.tokenizer import BasicSmilesTokenizer
import torch.nn as nn
import random
from collections import deque


def rotate_smiles(smiles,num): #,canonical=True,isomericSmiles=True):
    """Perform a rotation of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    try:
        ans = deque(list(range(m.GetNumAtoms())))
    except:
        print(f"Warning, could not rotate {smiles}")
        return None
    ans.rotate(num)
    nm = Chem.RenumberAtoms(m,ans)
    return Chem.MolToSmiles(nm, canonical=False)#, canonical=canonical, isomericSmiles=isomericSmiles)

RDLogger.DisableLog('rdApp.*')

def absolute_smiles(smiles):
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),canonical=True, isomericSmiles=True)
    except:
        print(f"Couldn't create absolute smiles for {smiles}")
        smiles = None
    return smiles


class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer
    
    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        leftpad: Add spaces to the left of the SMILES
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    """
    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        
    def fit(self, smiles, extra_chars=[], extra_pad = 5):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset
        
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        """
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad
        
    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot =  np.zeros((smiles.shape[0], self.pad, self._charlen),dtype=np.int8)
        
        if self.leftpad:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j,c in enumerate(ss):
                    one_hot[i,j+diff,self._char_to_int[c]] = 1
            return one_hot
        else:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                for j,c in enumerate(ss):
                    one_hot[i,j,self._char_to_int[c]] = 1
            return one_hot


class SmilesTransformations(nn.Module):
    def __init__(self,
                 mask_size=4,
                 **kwargs):
        super().__init__()
        self.mask_size = 4
        self.init_parser = BasicSmilesTokenizer()

    def _set_dummy(self, atom):
        return atom.SetAtomicNum(0)

    def _get_n_neighbors(self, seed_atom, n):
        "Recursive connected atom search"
        neighbors = seed_atom.GetNeighbors()
        ret_n = set()
        for ne in neighbors:
            ret_n.add(ne)
            if len(ret_n) == n:
                return ret_n
        for ne in ret_n:
            ret_n = ret_n.union(self._get_n_neighbors(ne, n=n - len(ret_n)))
            if len(ret_n) == n:
                return ret_n

    def mask(self, smiles, seed=-1):
        mol = Chem.MolFromSmiles(smiles)
        if seed < 0:
            seed = random.randint(0, len(mol.GetAtoms())-1)
        # Tag atoms and then rotate instead of set up dummy?
        assert seed <= len(mol.GetAtoms())
        seed_atom = tuple(mol.GetAtoms())[seed]
        atoms_to_mask = self._get_n_neighbors(seed_atom, self.mask_size)
        atom_ns = {atom.GetIdx() for atom in atoms_to_mask}
        for atom in mol.GetAtoms():
            if atom.GetIdx() in atom_ns:
                atom.SetAtomicNum(0)
        return Chem.MolToSmiles(mol, canonical=False)  # .replace(':','')

    def process(self, smiles):
        """
        N.B. We use * token here to additionally mask non-atom tokens
        The output is no longer a valid SMILES string and so no-longer
        Cleanly Rotatable.
        """
        parsed = self.init_parser.tokenize(smiles.replace(':', ''))
        parsed = ['*' if '*' in x else x for x in parsed]
        for i, j in zip(range(0, len(parsed) - 1), range(1, len(parsed))):
            if parsed[i] == '=' and parsed[j] == '*':
                parsed[i] = '*'
            elif parsed[i] == '*' and parsed[j].isnumeric():
                parsed[j] = '*'
        parsed = self._check_parentheses(parsed)
        return "".join(parsed)

    def _check_parentheses(self, parsed):
        for i in range(len(parsed)):
            if parsed[i] == '(':
                for j in range(i + 1, len(parsed)):
                    if parsed[j] == ')':
                        parsed[i] = '*'
                        parsed[j] = '*'
                        break
                    elif parsed[j] != '*':
                        break
        return parsed

    def rotate_smiles(self, smiles, num):
        """Perform a rotation of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        try:
            ans = deque(list(range(m.GetNumAtoms())))
        except Exception as e:
            print(e)
            print(f"Warning, could not rotate {smiles}")
            return 
        ans.rotate(num)
        nm = Chem.RenumberAtoms(Chem.Mol(m), ans)
        return Chem.MolToSmiles(nm, canonical=False)  # , canonical=canonical, isomericSmiles=isomericSmiles)

    def forward(self, smiles, rot, rot_init=0):
        assert isinstance(smiles, str), f"{type(smiles)} for smiles string"
        assert isinstance(rot, int), f"{type(rot)} for rotation integer"
        if rot_init > 0:
            smiles = self.rotate_smiles(smiles, rot_init)
        masked_smiles = self.mask(smiles)
        rotated_smiles = self.rotate_smiles(smiles, rot)
        rotated_masked_smiles = self.rotate_smiles(masked_smiles, rot)
        return smiles, self.process(masked_smiles), rotated_smiles, self.process(rotated_masked_smiles)


      
