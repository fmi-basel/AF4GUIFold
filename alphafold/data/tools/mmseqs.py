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
# MMSeqsAPI adapted from https://github.com/sokrypton/ColabFold (https://doi.org/10.5281/zenodo.5123296)

"""Library to run MMSeqs from Python."""

import glob
import os
import random
import re
import subprocess
import tarfile
import time
from typing import Any, List, Mapping, Optional, Sequence
import math
from shutil import copy, rmtree
from contextlib import contextmanager

from numpy import extract
from absl import logging
import requests
from alphafold.data.tools import utils
from alphafold.data.parsers import parse_fasta

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
                 db_load_mode: int = 0,
                 n_cpu: int = 1,
                 custom_tempdir=None,
                 use_index=False):
        """Initializes the Python MMSeqs wrapper.


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
        self.uniref90_db = self.databases[2]
        self.uniprot_db = self.databases[3]

        if use_index:
            self.uniref_db_suffix1 = f"{self.uniref30_db}.idx"
            self.uniref_db_suffix2 = f"{self.uniref30_db}.idx"
            self.env_db_suffix1 = f"{self.env_db}.idx"
            self.env_db_suffix2 = f"{self.env_db}.idx"
            self.uniref90_db_suffix1 = f"{self.uniref90_db}.idx"
            self.uniref90_db_suffix2 = f"{self.uniref90_db}.idx"
            self.uniprot_db_suffix1 = f"{self.uniprot_db}.idx"
            self.uniprot_db_suffix2 = f"{self.uniprot_db}.idx"
        else:
            self.uniref_db_suffix1 = f"{self.uniref30_db}_seq"
            self.uniref_db_suffix2 = f"{self.uniref30_db}_aln"
            self.env_db_suffix1 = f"{self.env_db}_seq"
            self.env_db_suffix2 = f"{self.env_db}_aln"
            self.uniref90_db_suffix1 = f"{self.uniref90_db}_seq"
            self.uniref90_db_suffix2 = f"{self.uniref90_db}_aln"
            self.uniprot_db_suffix1 = f"{self.uniprot_db}_seq"
            self.uniprot_db_suffix2 = f"{self.uniprot_db}_aln"
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

    def query(self, input_fasta_path: str, batch: bool = False, uniref90: bool = False, uniprot: bool = False) -> List[Mapping[str, Any]]:
        """Queries the database using MMSeqs."""
        logging.info("Databases:")
        logging.info(self.databases)
        logging.info(self.uniref90_db)
        a3m_dict = {}
        with utils.tmpdir_manager(self.custom_tempdir) as query_tmp_dir:
            output_a3m_path = os.path.join(query_tmp_dir, "0.a3m")
            output_packed_a3m_path = os.path.join(query_tmp_dir, 'output_packed.a3m')
            uniref_a3m_path = os.path.join(query_tmp_dir, 'uniref.a3m')
            env_a3m_path = os.path.join(query_tmp_dir, 'env.a3m')
            uniref90_a3m_path = os.path.join(query_tmp_dir, 'uniref90_hits.a3m')
            uniprot_a3m_path = os.path.join(query_tmp_dir, 'uniprot.a3m')
            qdb_path = os.path.join(query_tmp_dir, "qdb")
            qdb_lookup = os.path.join(query_tmp_dir, "qdb.lookup")

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
            search_param_uniref90_uniprot = ["--num-iterations", "3",
                                "--db-load-mode", self.db_load_mode,
                                "-a",
                                "-s", self.s,
                                "-e", "0.0001",
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
            
            stdout_list, stderr_list = [], []
            self.run_cmds([cmd_create_db], stdout_list, stderr_list)

            if not uniref90 and not uniprot:
                with utils.tmpdir_manager(self.custom_tempdir) as uniref_tmp_dir, \
                        utils.tmpdir_manager(self.custom_tempdir) as env_tmp_dir:
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


                    if batch:
                        with open(input_fasta_path) as f:
                            seqs, descs = parse_fasta(f.read())
                        with open(qdb_lookup, 'w') as f:
                            for i, desc in enumerate(descs):
                                f.write(f"{i}\t{desc}\t{i}\n")
                        with open(qdb_lookup) as f:
                            logging.info("Wrote qdb.lookup:")
                            logging.info(f.read())

                    
                    uniref_cmds = []
                    if self.db_load_mode == 3:
                        uniref_cmds.append(cmd_touchdb_uniref)
                    uniref_cmds.extend([cmd_uniref_search,
                            cmd_uniref_expandaln,
                            cmd_uniref_mvdb,
                            cmd_uniref_lndb,
                            cmd_uniref_align,
                            cmd_uniref_filterresult,
                            cmd_uniref_result2msa])

                    
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


                    env_cmds = []
                    if self.db_load_mode == 3:
                        env_cmds = [cmd_touchdb_env]
                    env_cmds.extend([cmd_env_search,
                            cmd_env_expandaln,
                            cmd_env_align,
                            cmd_env_filterresult,
                            cmd_env_result2msa,
                            cmd_merge_dbs,
                            cmd_unpack])
                    
                    self.run_cmds(env_cmds, stdout_list, stderr_list)


            if uniref90:
                logging.info("Running MMSeqs against uniref90 database")
                logging.info(f"Database path: {self.uniref90_db_suffix1}\nQuery temp dir: {qdb_path}")
                with utils.tmpdir_manager(self.custom_tempdir) as uniref90_tmp_dir:
                    cmd_touchdb_uniref90 = [self.binary_path,
                                "touchdb",
                                self.uniref90_db_suffix1]

                    cmd_uniref90_search = [self.binary_path,
                                                "search",
                                                    qdb_path,
                                                self.uniref90_db,
                                                os.path.join(uniref90_tmp_dir, "res"),
                                                os.path.join(uniref90_tmp_dir, "tmp"),
                                                "--threads", self.n_cpu,
                                                ] + search_param_uniref90_uniprot


                    cmd_uniref90_filterresult = [self.binary_path,
                                    "filterresult",
                                    qdb_path,
                                    self.uniref90_db_suffix1,
                                    os.path.join(uniref90_tmp_dir, "res"),
                                    os.path.join(uniref90_tmp_dir, "res_filter"),
                                    "--db-load-mode", self.db_load_mode,
                                    "--qid", "0",
                                    "--qsc", self.qsc,
                                    "--diff", "0",
                                    "--threads", self.n_cpu,
                                    "--max-seq-id", "1.0",
                                    "--filter-min-enable", "100"]

                    cmd_uniref90_result2msa = [self.binary_path,
                                    "result2msa",
                                    qdb_path,
                                    self.uniref90_db_suffix1,
                                    os.path.join(uniref90_tmp_dir, "res_filter"),
                                    uniref90_a3m_path,
                                    "--msa-format-mode", "6",
                                    "--db-load-mode", self.db_load_mode,
                                    "--threads", self.n_cpu] + filter_param
                    
                    cmd_uniref90_unpack = [self.binary_path,
                                "unpackdb",
                                uniref90_a3m_path,
                                query_tmp_dir,
                                "--unpack-name-mode", "0",
                                "--unpack-suffix", ".a3m"]
                    
                    uniref90_cmds = []
                    if self.db_load_mode == 3:
                        uniref90_cmds.append(cmd_touchdb_uniref90)
                    uniref90_cmds.extend([cmd_uniref90_search,
                            #cmd_uniref90_expandaln,
                            #cmd_uniref90_mvdb,
                            #cmd_uniref90_lndb,
                            #cmd_uniref90_align,
                            cmd_uniref90_filterresult,
                            cmd_uniref90_result2msa,
                            cmd_uniref90_unpack])
                    
                    logging.info("Uniref90 commands:")
                    logging.info(uniref90_cmds)
                    self.run_cmds(uniref90_cmds, stdout_list, stderr_list)


            if uniprot:
                with utils.tmpdir_manager(self.custom_tempdir) as uniprot_tmp_dir:
                    cmd_touchdb_uniprot = [self.binary_path,
                                "touchdb",
                                self.uniprot_db_suffix1]

                    cmd_uniprot_search = [self.binary_path,
                                                "search",
                                                    qdb_path,
                                                self.uniprot_db,
                                                os.path.join(uniprot_tmp_dir, "res"),
                                                os.path.join(uniprot_tmp_dir, "tmp"),
                                                "--threads", self.n_cpu,
                                                ] + search_param_uniref90_uniprot

                    cmd_uniprot_filterresult = [self.binary_path,
                                    "filterresult",
                                    qdb_path,
                                    self.uniprot_db_suffix1,
                                    os.path.join(uniprot_tmp_dir, "res"),
                                    os.path.join(uniprot_tmp_dir, "res_filter"),
                                    "--db-load-mode", self.db_load_mode,
                                    "--qid", "0",
                                    "--qsc", self.qsc,
                                    "--diff", "0",
                                    "--threads", self.n_cpu,
                                    "--max-seq-id", "1.0",
                                    "--filter-min-enable", "100"]

                    cmd_uniprot_result2msa = [self.binary_path,
                                    "result2msa",
                                    qdb_path,
                                    self.uniprot_db_suffix1,
                                    os.path.join(uniprot_tmp_dir, "res_filter"),
                                    uniprot_a3m_path,
                                    "--msa-format-mode", "6",
                                    "--db-load-mode", self.db_load_mode,
                                    "--threads", self.n_cpu] + filter_param
                    
                    cmd_uniprot_unpack = [self.binary_path,
                                "unpackdb",
                                uniprot_a3m_path,
                                query_tmp_dir,
                                "--unpack-name-mode", "0",
                                "--unpack-suffix", ".a3m"]
                    
                    uniprot_cmds = []
                    if self.db_load_mode == 3:
                        uniprot_cmds.append(cmd_touchdb_uniprot)
                    uniprot_cmds.extend([cmd_uniprot_search,
                            #cmd_uniprot_expandaln,
                            #cmd_uniprot_mvdb,
                            #cmd_uniprot_lndb,
                            #cmd_uniprot_align,
                            cmd_uniprot_filterresult,
                            cmd_uniprot_result2msa,
                            cmd_uniprot_unpack])
                    
                    self.run_cmds(uniprot_cmds, stdout_list, stderr_list)


            if batch:
                file_id_name_mapping = {}
                with open(qdb_lookup, 'r') as f:
                    for line in f.readlines():
                        id, name, file_id = line.split('\t')
                        file_id_name_mapping[name.strip('\n')] = file_id.strip('\n')

                for name, file_id in file_id_name_mapping.items():
                    logging.info("Query temp dir")
                    logging.info(os.listdir(query_tmp_dir))
                    with open(os.path.join(query_tmp_dir, f"{file_id}.a3m")) as f:
                        a3m_dict[name] = f.read()
            else:
                with open(output_a3m_path) as f:
                    a3m = f.read()

        stdout = '\n'.join(stdout_list)
        stderr = '\n'.join(stderr_list)

        if batch:
            return a3m_dict, stdout, stderr
        else:
            raw_output = dict(
                a3m=a3m,
                output=stdout,
                stderr=stderr)
            return [raw_output]
    

class MMSeqsAPI:
    """Adapted from https://github.com/sokrypton/ColabFold (https://doi.org/10.5281/zenodo.5123296)"""
    def __init__(self,
                 *,
                 host_url="https://api.colabfold.com",
                 custom_tempdir=None):
        self.host_url = host_url
        self.submission_endpoint = "ticket/msa"
        self.stdout_list, self.stderr_list = [], []
        self.custom_tempdir = custom_tempdir

    def submit(self, seqs: List[str], mode: str = 'env', seq_name: int = 101):
        query = "\n".join([f">{seq_name}\n{seq}" for seq_name, seq in enumerate(seqs, seq_name)])
        error_count = 0
        logging.info(f"Submitting {query} to Colabfold MSA server...")

        while True:
            try:
                res = requests.post(f'{self.host_url}/{self.submission_endpoint}', data={'q': query, 'mode': mode}, timeout=6.02)
                res.raise_for_status()
                break
            except requests.exceptions.Timeout:
                warn_msg = "Timeout while submitting to MSA server. Retrying..."
                self.stdout_list.append(warn_msg)
                logging.warning(warn_msg)
            except Exception as e:
                error_count += 1
                warn_msg = f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                self.stdout_list.append(warn_msg)
                logging.warning(warn_msg)
                warn_msg = f"Error: {e}"
                self.stdout_list.append(warn_msg)
                logging.warning(warn_msg)
                time.sleep(5)
                if error_count > 5:
                    raise
        try:
            result = res.json()
        except ValueError:
            error_msg = f"Server didn't reply with JSON: {res.text}"
            self.stderr_list.append(error_msg)
            logging.error(error_msg)
            result = {"status": "ERROR"}
        return result
    
    def get_status(self, ID: str):
        while True:
            error_count = 0
            try:
                res = requests.get(f'{self.host_url}/ticket/{ID}', timeout=6.02)
                res.raise_for_status()
                break
            except requests.exceptions.Timeout:
                warn_msg = "Timeout while fetching status from MSA server. Retrying..."
                self.stdout_list.append(warn_msg)
                logging.warning(warn_msg)
            except Exception as e:
                error_count += 1
                warn_msg = f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                self.stdout_list.append(warn_msg)
                logging.warning(warn_msg)
                warn_msg = f"Error: {e}"
                self.stdout_list.append(warn_msg)
                logging.warning(warn_msg)
                time.sleep(5)
                if error_count > 5:
                    raise
        try:
            result = res.json()
        except ValueError:
            error_msg = f"Server didn't reply with JSON: {res.text}"
            self.stderr_list.append(error_msg)
            logging.error(error_msg)
            result = {"status": "ERROR"}
        return result
    
    def download(self, ID: str):
        error_count = 0
        merged_output = []
        while True:
            try:
                res = requests.get(f'{self.host_url}/result/download/{ID}', timeout=6.02)
                res.raise_for_status()
                break
            except requests.exceptions.Timeout:
                warn_msg = "Timeout while fetching result from MSA server. Retrying..."
                self.stdout_list.append(warn_msg)
                logging.warning("Timeout while fetching result from MSA server. Retrying...")
            except Exception as e:
                error_count += 1
                warn_msg = f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                self.stdout_list.append(warn_msg)
                logging.warning(warn_msg)
                warn_msg = f"Error: {e}"
                self.stdout_list.append(warn_msg)
                logging.warning(warn_msg)
                time.sleep(5)
                if error_count > 5:
                    raise
        with utils.tmpdir_manager(os.path.join(self.custom_tempdir)) as extract_tmp_dir:
            temp_tar_gz = os.path.join(extract_tmp_dir, 'mmseqs2_output.tar.gz')
            temp_tar_gz_extracted = os.path.join(extract_tmp_dir, 'mmseqs2_output')
            with open(temp_tar_gz, 'wb') as f:
                f.write(res.content)
            with tarfile.open(temp_tar_gz, 'r') as tar_gz:
                tar_gz.extractall(temp_tar_gz_extracted)
            files = os.listdir(temp_tar_gz_extracted)
            expected_files = ['bfd.mgnify30.metaeuk30.smag30.a3m', 'uniref.a3m']
            if not expected_files[0] in files or not expected_files[1] in files:
                error_msg = f"Downloaded MSA archive does not contain expected files: {expected_files}. It contains: {files}"
                self.stdout_list.append(error_msg)
                raise Exception(error_msg)
            for file in [os.path.join(temp_tar_gz_extracted, file) for file in files]:
                for expected_file in expected_files:
                    if file.endswith(expected_file):
                        with open(file, 'r') as f:
                            output = f.read()
                        merged_output.append(output)
        return '\n'.join(merged_output)

    def parse_output_a3m(self, output) -> str:
        """Remove Xs from the start and end of the sequence lines. Remove empty lines and remove \x00 characters. Return a string with lines separated by \n."""
        x_pos = []
        new_lines = []
        lines = output.split("\n")
        input_seq_name = lines[0]
        input_seq = lines[1]
        pattern = r'(^X*)(.*?[^X])(X*$)'

        matches = re.match(pattern, input_seq)

        logging.debug(f"Matches: {matches}")
        if matches:
            len_match_start = len(matches.group(1))
            len_match_end = len(matches.group(3))
        if len_match_start == 0:
            len_match_start = None
        if len_match_end == 0:
            len_match_end = None
        logging.debug(f"len_match_start: {len_match_start}, len_match_end: {len_match_end}")
        input_seq = ''.join(list(lines[1])[len_match_start:-len_match_end])
        for i, line in enumerate(lines):
            if line == '\n':
                logging.debug(f"Empty line (number {i}) found in output. Skipping...")
                continue    
            if "\x00" in line:
                line = line.replace("\x00","")
            if not line.startswith(">"):
                line = list(line)
                line = line[len_match_start:-len_match_end]
                line = ''.join(line)
            if line:
                #if (input_seq_name == line or input_seq == line) and i > 1:
                #    logging.debug(f"Input seq_name or input_seq found in line {i} of output. Skipping...")
                #    continue
                new_lines.append(line)
        return '\n'.join(new_lines)
            
    def query(self, input_fasta_path: str) -> List[Mapping[str, Any]]:
        with open(input_fasta_path, 'r') as f:
            input_sequence = parse_fasta(f.read())[0]
        if len(input_sequence) != 1:
            error_msg = f"Input FASTA file contains {len(input_sequence)} sequences, but only one sequence is currently supported."
            self.stderr_list.append(error_msg)
            raise ValueError(error_msg)

        while True:
            # Resubmit job until it goes through
            result = self.submit([input_sequence])
            while result["status"] in ["UNKNOWN", "RATELIMIT"]:
                sleep_time = 5 + random.randint(0, 5)
                logging.error(f"Sleeping for {sleep_time}s. Reason: {result['status']}")
                # resubmit
                time.sleep(sleep_time)
                result = self.submit(input_sequence)
            if result["status"] == "ERROR":
                error_msg = f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.'
                self.stderr_list.append(error_msg)
                raise Exception(error_msg)
            if result["status"] == "MAINTENANCE":
                error_msg = f"MMseqs2 API is undergoing maintenance. Please try again in a few minutes."
                self.stderr_list.append(error_msg)
                raise Exception(error_msg)

            # wait for job to finish
            job_id,TIME = result["id"],0

            while result["status"] in ["UNKNOWN","RUNNING","PENDING"]:
                t = 5 + random.randint(0,5)
                logging.info(f"MMseqs2 job is {result['status']}. Waiting for {t}s before checking status again.")
                time.sleep(t)
                result = self.get_status(job_id)
            if result["status"] == "COMPLETE":
                break
            if result["status"] == "ERROR":
                error_msg = f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.'
                self.stderr_list.append(error_msg)
                raise Exception(error_msg)

        # download results
        output = self.download(job_id)

        a3m = self.parse_output_a3m(output)

        stdout = '\n'.join(self.stdout_list)
        stderr = '\n'.join(self.stderr_list)
        
        raw_output = dict(
            a3m=a3m,
            output=stdout,
            stderr=stderr)
        return [raw_output]