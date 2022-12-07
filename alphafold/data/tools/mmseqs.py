# Copyright 2022 Friedrich Miescher Institute for Biomedical Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Georg Kempf, Friedrich Miescher Institute for Biomedical Research
# MMSeqs workflow and parameters adapted from https://github.com/sokrypton/ColabFold (https://doi.org/10.5281/zenodo.5123296)

"""Library to run MMSeqs from Python."""

import glob
import os
import subprocess
from typing import Any, List, Mapping, Optional, Sequence
import math
from shutil import copy, rmtree
from contextlib import contextmanager
from absl import logging
from alphafold.data.tools import utils


class MMSeqs:
    """Python wrapper of the MMSeqs2 workflow."""

    def __init__(self,
                 *,
                 binary_path: str,
                 database_path: Sequence[str],
                 filter: bool = True,
                 expand_eval: float = math.inf,
                 align_eval: int = 10,
                 diff: int = 3000,
                 qsc: float = -20.0,
                 max_accept: int = 1000000,
                 s: float = 8,
                 db_load_mode: int = 3,
                 n_cpu: int = 1,
                 custom_tempdir=None,
                 use_index=False):
        """Initializes the Python MMSeqs wrapper.

        Args:
          binary_path: The path to the MMSeqs executable.
          databases: A sequence of MMSeqs database paths. This should be the
            common prefix for the database files (i.e. up to but not including
            _hhm.ffindex etc.)
          n_cpu: The number of CPUs to give MMSeqs.
          n_iter: The number of MMSeqs iterations.
          e_value: The E-value, see MMSeqs docs for more details.
          maxseq: The maximum number of rows in an input alignment. Note that this
            parameter is only supported in MMSeqs version 3.1 and higher.
          realign_max: Max number of HMM-HMM hits to realign. MMSeqs default: 500.
          maxfilt: Max number of hits allowed to pass the 2nd prefilter.
            MMSeqs default: 20000.
          min_prefilter_hits: Min number of hits to pass prefilter.
            MMSeqs default: 100.
          all_seqs: Return all sequences in the MSA / Do not filter the result MSA.
            MMSeqs default: False.
          alt: Show up to this many alternative alignments.
          p: Minimum Prob for a hit to be included in the output hhr file.
            MMSeqs default: 20.
          z: Hard cap on number of hits reported in the hhr file.
            MMSeqs default: 500. NB: The relevant MMSeqs flag is -Z not -z.

        Raises:
          RuntimeError: If MMSeqs binary not found within the path.
        """
        self.binary_path = binary_path
        self.databases = database_path

        #for database_path in self.databases:
        #    if not glob.glob(database_path + '_*'):
        #        logging.error('Could not find MMSeqs2 database %s', database_path)
        #        #raise ValueError(f'Could not find MMSeqs database {database_path}')
        self.uniref30_db = self.databases[0]
        self.env_db = self.databases[1]
        if use_index:
            self.uniref_db_suffix1 = f"{self.uniref30_db}.idx"
            self.uniref_db_suffix2 = f"{self.uniref30_db}.idx"
            self.env_db_suffix1 = f"{self.env_db}.idx"
            self.env_db_suffix2 = f"{self.env_db}.idx"
        else:
            self.uniref_db_suffix1 = f"{self.uniref30_db}_seq"
            self.uniref_db_suffix2 = f"{self.uniref30_db}_aln"
            self.env_db_suffix1 = f"{self.env_db}_seq"
            self.env_db_suffix2 = f"{self.env_db}_aln"
        self.filter = filter
        self.expand_eval = expand_eval
        self.align_eval = align_eval
        self.diff = diff
        self.qsc = qsc
        self.max_accept = max_accept
        self.s = s
        self.db_load_mode = db_load_mode
        self.n_cpu = n_cpu
        self.custom_tempdir = custom_tempdir

    def run_cmds(self, cmds, stdout_list, stderr_list):
        for cmd in cmds:
            for i, item in enumerate(cmd):
                if type(item) == bool:
                    if item:
                        cmd[i] = "1"
                    else:
                        cmd[i] = "0"
                else:
                    cmd[i] = str(item)
            logging.info(f"Launching subprocess {' '.join(cmd)} with {self.n_cpu} threads.")
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with utils.timing('mmseqs2 query'):
                stdout, stderr = process.communicate()
                stdout_list.append(stdout.decode('utf-8'))
                stderr_list.append(stderr.decode('utf-8'))
                retcode = process.wait()

            if retcode:
                # Logs have a 15k character limit, so log MMSeqs blits error line by line.
                logging.error('MMSeqs2 failed. MMSeqs2 stderr begin:')
                for error_line in stderr.decode('utf-8').splitlines():
                    if error_line.strip():
                        logging.error(error_line.strip())
                logging.error('MMSeqs2 stderr end')
                raise RuntimeError('MMSeqs2 failed\nstdout:\n%s\n\nstderr:\n%s\n' % (
                    stdout.decode('utf-8'), stderr[:500_000].decode('utf-8')))

    def query(self, input_fasta_path: str) -> List[Mapping[str, Any]]:
        """Queries the database using MMSeqs."""
        with utils.tmpdir_manager(os.path.join(self.custom_tempdir, "query")) as query_tmp_dir, \
                utils.tmpdir_manager(os.path.join(self.custom_tempdir, "uniref")) as uniref_tmp_dir, \
                utils.tmpdir_manager(os.path.join(self.custom_tempdir, "env")) as env_tmp_dir:
            output_a3m_path = os.path.join(query_tmp_dir, "0.a3m")
            output_packed_a3m_path = os.path.join(query_tmp_dir, 'output_packed.a3m')
            uniref_a3m_path = os.path.join(query_tmp_dir, 'uniref.a3m')
            env_a3m_path = os.path.join(query_tmp_dir, 'env.a3m')
            qdb_path = os.path.join(query_tmp_dir, "qdb")

            if self.filter:
                # 0.1 was not used in benchmarks due to POSIX shell bug in line above
                #  EXPAND_EVAL=0.1
                self.align_eval = 10
                self.qsc = 0.8
                self.max_accept = 100000

            search_param = ["--num-iterations", "3",
                                "--db-load-mode", self.db_load_mode,
                                "-a",
                                "-s", self.s,
                                "-e", "0.1",
                                "--max-seqs", "10000",]
            filter_param = ["--filter-msa", self.filter,
                            "--filter-min-enable", "1000",
                            "--diff", self.diff,
                            "--qid", "0.0,0.2,0.4,0.6,0.8,1.0",
                            "--qsc", "0",
                            "--max-seq-id", "0.95",]
            expand_param = ["--expansion-mode", "0",
                            "-e", self.expand_eval,
                            "--expand-filter-clusters", self.filter,
                            "--max-seq-id", "0.95",]

            cmd_create_db = [self.binary_path,
                            "createdb",
                             input_fasta_path,
                            qdb_path,
                            "--shuffle", "0"]

            cmd_touchdb_uniref = [self.binary_path,
                           "touchdb",
                           self.uniref_db_suffix1]


            cmd_uniref_search = [self.binary_path,
                   "search",
                    qdb_path,
                   self.uniref30_db,
                   os.path.join(uniref_tmp_dir, "res"),
                   os.path.join(uniref_tmp_dir, "tmp"),
                   "--threads", self.n_cpu,
                   ] + search_param

            cmd_uniref_expandaln = [self.binary_path,
                                    "expandaln",
                                    qdb_path,
                                    self.uniref_db_suffix1,
                                    os.path.join(uniref_tmp_dir, "res"),
                                    self.uniref_db_suffix2,
                                    os.path.join(uniref_tmp_dir, "res_exp"),
                                    "--db-load-mode", self.db_load_mode,
                                    "--threads", self.n_cpu,
                                    ] + expand_param
            cmd_uniref_mvdb = [self.binary_path,
                               "mvdb",
                                os.path.join(uniref_tmp_dir, "tmp", "latest", "profile_1"),
                                os.path.join(uniref_tmp_dir, "prof_res")]
            cmd_uniref_lndb = [self.binary_path,
                               "lndb",
                                os.path.join(uniref_tmp_dir, "qdb_h"),
                                os.path.join(uniref_tmp_dir, "prof_res_h")]

            cmd_uniref_align = [self.binary_path,
                                "align",
                                 os.path.join(uniref_tmp_dir, "prof_res"),
                                 self.uniref_db_suffix1,
                                 os.path.join(uniref_tmp_dir, "res_exp"),
                                 os.path.join(uniref_tmp_dir, "res_exp_realign"),
                                 "--db-load-mode", self.db_load_mode,
                                 "-e", self.align_eval,
                                 "--max-accept", self.max_accept,
                                 "--threads", self.n_cpu,
                                 "--alt-ali", "10",
                                 "-a"]
            cmd_uniref_filterresult = [self.binary_path,
                                        "filterresult",
                                        qdb_path,
                                        self.uniref_db_suffix1,
                                        os.path.join(uniref_tmp_dir, "res_exp_realign"),
                                        os.path.join(uniref_tmp_dir, "res_exp_realign_filter"),
                                        "--db-load-mode", self.db_load_mode,
                                        "--qid", "0",
                                        "--qsc", self.qsc,
                                        "--diff", "0",
                                        "--threads", self.n_cpu,
                                        "--max-seq-id", "1.0",
                                        "--filter-min-enable", "100"]

            cmd_uniref_result2msa = [self.binary_path,
                                        "result2msa",
                                        qdb_path,
                                        self.uniref_db_suffix1,
                                        os.path.join(uniref_tmp_dir, "res_exp_realign_filter"),
                                        uniref_a3m_path,
                                        "--msa-format-mode", "6",
                                        "--db-load-mode", self.db_load_mode,
                                        "--threads", self.n_cpu] + filter_param

            cmd_touchdb_env = [self.binary_path,
                           "touchdb",
                           self.env_db_suffix1]

            uniref_cmds = [cmd_create_db,
                    #cmd_touchdb_uniref,
                    cmd_uniref_search,
                    cmd_uniref_expandaln,
                    cmd_uniref_mvdb,
                    cmd_uniref_lndb,
                    cmd_uniref_align,
                    cmd_uniref_filterresult,
                    cmd_uniref_result2msa]

            stdout_list, stderr_list = [], []
            self.run_cmds(uniref_cmds, stdout_list, stderr_list)

            cmd_env_search = [self.binary_path,
                              "search",
                              os.path.join(uniref_tmp_dir, "prof_res"),
                              self.env_db,
                              os.path.join(env_tmp_dir, "res_env"),
                              os.path.join(env_tmp_dir, "tmp"),
                              "--threads", self.n_cpu] + search_param

            cmd_env_expandaln = [self.binary_path,
                                 "expandaln",
                                 os.path.join(uniref_tmp_dir, "prof_res"),
                                 self.env_db_suffix1,
                                 os.path.join(env_tmp_dir, "res_env"),
                                 self.env_db_suffix2,
                                 os.path.join(env_tmp_dir, "res_env_exp"),
                                 "-e", self.expand_eval,
                                 "--expansion-mode", "0",
                                 "--db-load-mode", self.db_load_mode,
                                 "--threads", self.n_cpu]

            cmd_env_align = [self.binary_path,
                             "align",
                             os.path.join(env_tmp_dir, "tmp", "latest", "profile_1"),
                             self.env_db_suffix1,
                             os.path.join(env_tmp_dir, "res_env_exp"),
                             os.path.join(env_tmp_dir, "res_env_exp_realign"),
                             "--db-load-mode", self.db_load_mode,
                             "-e", self.align_eval,
                             "--max-accept", self.max_accept,
                             "--threads", self.n_cpu,
                             "--alt-ali", "10",
                             "-a"]

            cmd_env_filterresult = [self.binary_path,
                                    "filterresult",
                                    qdb_path,
                                    self.env_db_suffix1,
                                    os.path.join(env_tmp_dir, "res_env_exp_realign"),
                                    os.path.join(env_tmp_dir, "res_env_exp_realign_filter"),
                                    "--db-load-mode", self.db_load_mode,
                                    "--qid", "0",
                                    "--qsc", self.qsc,
                                    "--diff", "0",
                                    "--max-seq-id", "1.0",
                                    "--threads", self.n_cpu,
                                    "--filter-min-enable", "100"]

            cmd_env_result2msa = [self.binary_path,
                                  "result2msa",
                                  qdb_path,
                                  self.env_db_suffix1,
                                  os.path.join(env_tmp_dir, "res_env_exp_realign_filter"),
                                  env_a3m_path,
                                  "--msa-format-mode", "6",
                                  "--db-load-mode", self.db_load_mode,
                                  "--threads", self.n_cpu] + filter_param

            cmd_merge_dbs = [self.binary_path,
                             "mergedbs",
                             qdb_path,
                             output_packed_a3m_path,
                             uniref_a3m_path,
                             env_a3m_path]

            cmd_unpack = [self.binary_path,
                          "unpackdb",
                          output_packed_a3m_path,
                          query_tmp_dir,
                          "--unpack-name-mode", "0",
                          "--unpack-suffix", ".a3m"]

            env_cmds = [
                    #cmd_touchdb_env,
                    cmd_env_search,
                    cmd_env_expandaln,
                    cmd_env_align,
                    cmd_env_filterresult,
                    cmd_env_result2msa,
                    cmd_merge_dbs,
                    cmd_unpack]
            self.run_cmds(env_cmds, stdout_list, stderr_list)

            with open(output_a3m_path) as f:
                a3m = f.read()

        stdout = '\n'.join(stdout_list)
        stderr = '\n'.join(stderr_list)

        raw_output = dict(
            a3m=a3m,
            output=stdout,
            stderr=stderr)
        return [raw_output]