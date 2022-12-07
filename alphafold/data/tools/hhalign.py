# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 Friedrich Miescher Institute for Biomedical Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Georg Kempf, Friedrich Miescher Institute for Biomedical Research

"""Library to run HHalign from Python."""

import glob
import os
import subprocess
from typing import Sequence

from absl import logging

from alphafold.data import parsers
from alphafold.data.tools import utils
# Internal import (7716).


class HHAlign:
  """Python wrapper of the HHsearch binary."""

  def __init__(self,
               binary_path: str):
    """Initializes the Python HHalign wrapper.

    Args:
      binary_path: The path to the HHsearch executable.

    Raises:
      RuntimeError: If HHalign binary not found within the path.
    """
    self.binary_path = binary_path

  @property
  def output_format(self) -> str:
    return 'hhr'

  @property
  def input_format(self) -> str:
    return 'a3m'

  def query(self, custom_template: str, a3m: str) -> str:
    """Queries the database using HHsearch using a given a3m."""
    #with utils.tmpdir_manager() as query_tmp_dir:
    query_tmp_dir = ""
    input_path = os.path.join(query_tmp_dir, 'query.a3m')
    hhr_path = os.path.join(query_tmp_dir, 'output.hhr')
    template_sequence_path = os.path.join(query_tmp_dir, 'template_sequence.fasta')
    with open(input_path, 'w') as f:
      f.write(a3m)
    with open(template_sequence_path, 'w') as f:
      f.write(f">{custom_template['name']}_A\n")
      f.write(custom_template['sequence'])

    cmd = [self.binary_path,
           '-i', input_path,
           '-t', template_sequence_path,
           '-o', hhr_path
           ]

    logging.info('Launching subprocess "%s"', ' '.join(cmd))
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with utils.timing('HHalign query'):
      stdout, stderr = process.communicate()
      retcode = process.wait()

    if retcode:
      # Stderr is truncated to prevent proto size errors in Beam.
      raise RuntimeError(
          'HHalign failed:\nstdout:\n%s\n\nstderr:\n%s\n' % (
              stdout.decode('utf-8'), stderr[:100_000].decode('utf-8')))

    with open(hhr_path) as f:
      hhr = f.read()
    return hhr

  def get_template_hits(self,
                        output_string: str,
                        input_sequence: str) -> Sequence[parsers.TemplateHit]:
    """Gets parsed template hits from the raw string output by the tool."""
    del input_sequence  # Used by hmmseach but not needed for hhsearch.
    return parsers.parse_hhr(output_string)
