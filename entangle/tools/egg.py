# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Modifications Copyright (c) 2025 [Zhanghan Wang]
# Note: Support better logging.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import os.path as osp
import shutil
import tempfile
from datetime import datetime
from subprocess import DEVNULL, PIPE, STDOUT, Popen

import rich
from entangle.sgraph.egraph import CannotFindPostconditions, EGraph
from entangle.sgraph.sexpr import ShapeLike
from entangle.utils.print_utils import BGREEN, BRED, BRI, RST, EntangleLogger, filling_terminal, get_global_logger, get_logger

LOGGER = None


class FailedImplyingEquivalence(Exception):
    pass


class FailedProveSelf(Exception):
    pass


class EggError(Exception):
    pass


class EggRunner:
    def __init__(
        self,
        egg_dirname: str = None,
        egg_data_dirname: str = None,
        tmux_session_name="infer",
        inverse_lemma=False,
        post_type="yis",
        log_level=logging.INFO,
        debug=False,
        tmux=False,
        verbose=False,
        visualize=False,
        stats=False,
        use_local_directory=False,
    ):
        assert egg_dirname is not None, "The directory to Egg is not specified."
        self.egg_dirname = egg_dirname
        self.egg_data_dirname = egg_data_dirname or f"{self.egg_dirname}/target/precondition"
        self.tmux_session_name = tmux_session_name
        self.inverse_lemma = inverse_lemma
        self.post_type = post_type
        self.log_level = log_level
        self.debug = debug
        self.tmux = tmux
        if self.tmux and os.name == "nt":
            raise ValueError("tmux mode is not supported on Windows.")
        self.verbose = verbose
        self.stats = stats
        self.visualize = visualize
        self.use_local_directory = use_local_directory

        global LOGGER
        LOGGER = get_global_logger()

        LOGGER.info(f"{BGREEN}Building Egg from {self.egg_dirname} ...{RST}")
        cmd = f"cd {self.egg_dirname} && {'cargo build' if self.debug else 'cargo build --release'}"
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
        p.wait()
        if p.returncode != 0:
            out, _ = p.communicate()
            LOGGER.info(out.decode())
            raise RuntimeError(f"Failed to build Egg (returncode={p.returncode}).")
        LOGGER.info(f"{BGREEN}Egg built.{RST}")
        if tmux:
            LOGGER.info(f"{BGREEN}Start tmux...{RST}")
            cmd = f"tmux kill-session -t {self.tmux_session_name}; tmux new-session -s {self.tmux_session_name} -n home -d"
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
            p.wait()
            if p.returncode != 0:
                out, _ = p.communicate()
                LOGGER.info(out.decode())
                raise RuntimeError(f"Failed to setup tmux (returncode={p.returncode}).")

        self.all_scalar_econd_str: str = None

    def clean(self, dirname, keep_root=False):
        if keep_root:
            os.system(f"rm -rf {dirname}/*")
        else:
            os.system(f"rm -rf {dirname}")

    def upload(self, dirname):
        if self.use_local_directory:
            LOGGER.info(f"{BGREEN}No need to upload: {dirname} ...{RST}")
            return
        LOGGER.info(f"{BGREEN}Uploading precondition and computation from {dirname}...{RST}")
        # Clean up precondition directory.
        cmd = f"rm -rf {self.egg_data_dirname}; mkdir -p {self.egg_data_dirname}"
        Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT).wait()

        # Copy precondtions
        cmd = f"cp -r {dirname}/* {self.egg_data_dirname}"
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
        p.wait()
        if p.returncode != 0:
            out, _ = p.communicate()
            LOGGER.info(out.decode())
            raise RuntimeError(f"Failed to upload (returncode={p.returncode}).")

    @staticmethod
    def get_output_log_path(dirname: str) -> str:
        return osp.symabspath(osp.join(dirname, "output.log"))

    def run(
        self,
        dirname: str,
        tmux_window_id: str,
        mode: str,
        additional_args: str = None,
        only_log: bool = True,
        silent: bool = False,
    ):
        """
        mode: can be "infer", "pass" or "check_impl".
            - "infer": run Egg to infer post conditions.
            - "pass": precondition pass through.
            - "check_impl": check if the implementation is correct.
        """
        if self.use_local_directory:
            egg_data_dirname = osp.symabspath(dirname)
        else:
            egg_data_dirname = self.egg_data_dirname
        output_log_path = self.get_output_log_path(egg_data_dirname)
        os.makedirs(egg_data_dirname, exist_ok=True)

        if mode == "infer":
            logger = get_logger(output_log_path, self.log_level, mode="a", path=output_log_path)
            self.infer_post_condition(
                egg_data_dirname,
                output_log_path=output_log_path,
                inverse_lemma=self.inverse_lemma,
                debug=self.debug,
                verbose=self.verbose,
                visualize=self.visualize,
                stats=self.stats,
                post_type=self.post_type,
                only_log=only_log,
                logger=logger,
            )
            self.check_result(dirname)
            return

        if mode == "check_impl":
            core_cmd = f"target/{'debug' if self.debug else 'release'}/egger --mode check_impl --input_dirname {egg_data_dirname} --saturated_path {egg_data_dirname}/saturated.json"
            grep_success_str = "Passed equivalence checking"
        elif mode == "self_provable":
            core_cmd = f"target/{'debug' if self.debug else 'release'}/egger --mode self_provable --input_dirname {egg_data_dirname} --saturated_path {egg_data_dirname}/saturated.json"
            grep_success_str = "Done self provable checking."
        else:
            raise ValueError(f"Unknown mode for egg.run: {mode}")
        # Add potential verbose
        if self.inverse_lemma:
            core_cmd += "--inverse_lemma "
        if self.verbose:
            core_cmd += "--verbose "
        if self.visualize:
            core_cmd += "--visualize "
        # Add additional arguments
        if additional_args is not None:
            core_cmd += f"{additional_args} "
        core_cmd = " export RUST_BACKTRACE=1; " + core_cmd
        if not self.tmux:
            with open(osp.join(dirname, "cmd.sh"), "w") as f:
                f.write(core_cmd)
            if only_log:
                cmd = f"cd {self.egg_dirname} && {core_cmd} 2>&1 | tee {output_log_path} > /dev/null"
            else:
                cmd = f"cd {self.egg_dirname} && {core_cmd} 2>&1 | tee {output_log_path}"
            # Check
            cmd += f" ; cat {output_log_path} | grep '{grep_success_str}'"
            p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
            if not silent:
                LOGGER.info(f"{BGREEN}Running Egg to {mode}...{RST}")
        else:
            core_cmd = core_cmd.strip(" ") + f" 2>&1 | tee {output_log_path} "
            tmux_session_window_name = f"{self.tmux_session_name}:{tmux_window_id}"
            cmd = (
                f"cd {self.egg_dirname} && tmux new-window -t {self.tmux_session_name} -n {tmux_window_id} -d && "
                # f"tmux pipe-pane -t {tmux_session_window_name} 'cat > {output_log_path}' && "
                f"""tmux send -t {tmux_session_window_name} "{core_cmd}" C-m "tmux wait -S 0" C-m; """
                f"""echo "Command sent to tmux, waiting..."; """
                f"tmux wait 0;"
                f"sleep 4;"
                f"tmux pipe-pane -t {tmux_session_window_name} '{output_log_path}'; "
                f"cat {output_log_path} | grep '{grep_success_str}'"
                # f"""tmux capture-pane -pS -100 -t {tmux_session_window_name} | grep "Succeeded in finding post conditions." """
            )
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
            if not silent:
                LOGGER.info(
                    f"{BGREEN}Running Egg to {mode}...{RST} "
                    f"(Use tmux window {BRI}{tmux_session_window_name}{RST} to see what happened.)"
                )
        p.wait()
        self.check_result(dirname)
        if p.returncode != 0:
            out, _ = p.communicate()
            if out is not None:
                LOGGER.info(filling_terminal("server output"))
                out = out.decode()
                if out.endswith("\n\n"):
                    out = out[:-1]
                LOGGER.info(out)
                LOGGER.info("\n" + filling_terminal())
            if self.tmux and mode != "pass":
                LOGGER.info(filling_terminal("tmux buffer output"))
                cmd = f"cat {output_log_path}"
                Popen(cmd, shell=True).wait()
                LOGGER.info("\n" + filling_terminal())
            self.download(dirname)
            if mode != "pass":
                raise CannotFindPostconditions(
                    f"Egg returns {p.returncode}, please check the log {output_log_path}\n{core_cmd}"
                )

    def infer_post_condition(
        self,
        precondition_dirname,
        output_log_path=None,
        saturated_path=None,
        output_path=None,
        inverse_lemma=False,
        iter_limit=40,
        node_limit=200000,
        time_limit=3600,
        explanation=False,
        debug=False,
        verbose=False,
        visualize=False,
        stats=False,
        post_type=None,
        only_log=True,
        logger: EntangleLogger = None,
        **kwargs,
    ):
        saturated_path = saturated_path or osp.join(precondition_dirname, "saturated.json")
        output_path = output_path or osp.join(precondition_dirname, "postcondition.sexpr")
        saturation_cmd = f"cd {self.egg_dirname} && " + (
            ("./target/release/egger " if not debug else "./target/debug/egger ")
            + ("--verbose " if verbose else "")
            + ("--visualize " if visualize else "")
            + ("--stats " if stats else "")
            + ("--explanation " if explanation else "")
            + f"""--mode saturate_precondition """
            + f"""--input_dirname {precondition_dirname} """
            + f"""--saturated_path {saturated_path} """
            + f"""--n_iter {iter_limit} --n_sec {time_limit} --n_nodes {node_limit} """
            + (f"""--inverse_lemma """ if inverse_lemma else "")
        )
        if self.debug:
            saturation_cmd = "export RUST_BACKTRACE=1;" + saturation_cmd
        if output_log_path is not None:
            if only_log:
                saturation_cmd += f" 2>&1 | tee {output_log_path} > /dev/null"
            else:
                saturation_cmd += f" 2>&1 | tee {output_log_path}"
        logger.print(saturation_cmd)
        begin = datetime.now()
        saturation_process = Popen(
            saturation_cmd,
            shell=True,
            stderr=STDOUT,
        )
        saturation_process.wait()
        returncode = saturation_process.returncode
        logger.print(f"Saturation done in {datetime.now() - begin}, returncode={returncode}")
        if returncode == 0:
            logger.print(f"{BRI}Saturation process succeeded.{RST}")
            begin = datetime.now()
            logger.print(f"output_log_path: {output_log_path}")
            if not osp.exists(saturated_path):
                raise RuntimeError(f"Cannot find saturated.json, please check {output_log_path}")
            egraph = EGraph(saturated_path, verbose=verbose, logger=logger)
            if visualize:
                egraph.visualize_saturated()
                valid_only = egraph.extract_to_valid_only()
                valid_only.to_dot(osp.join(precondition_dirname, "valid_only.dot"))
            egraph.compute_post_condition_eclasses()

            candidates_egraph = None
            if visualize:
                candidates_egraph = egraph.extract_to_postcondition(all_candidates=True)
                candidates_egraph.to_dot(osp.join(precondition_dirname, "candidates.dot"))
                postcondition_egraph = egraph.extract_to_postcondition()
                postcondition_egraph.to_dot(osp.join(precondition_dirname, "postcondition.dot"))

            def get_representative_egraph(egraph: EGraph, visualize: bool) -> EGraph:
                postcondition_representative_egraph = egraph.extract_to_postcondition(representative_only=True)
                if visualize:
                    file_name = "postcondition_representative.dot"
                    path = osp.join(precondition_dirname, file_name)
                    postcondition_representative_egraph.to_dot(path)
                return postcondition_representative_egraph

            def get_yis_egraph(egraph: EGraph, visualize: bool) -> EGraph:
                postcondition_representative_egraph = egraph.extract_to_postcondition(
                    representative_only=True,
                    including_yis=True,
                )
                if visualize:
                    file_name = "postcondition_including_yis.dot"
                    path = osp.join(precondition_dirname, file_name)
                    postcondition_representative_egraph.to_dot(path)
                return postcondition_representative_egraph

            # used_egraph = postcondition_including_yis_egraph
            if post_type is None:
                postcondition_representative_egraph = get_representative_egraph(egraph, visualize)
                if postcondition_representative_egraph.all_yi_included():
                    used_egraph = postcondition_representative_egraph
                else:
                    if candidates_egraph is not None:
                        used_egraph = candidates_egraph
                    else:
                        used_egraph = egraph.extract_to_postcondition(all_candidates=True)
            else:
                if post_type == "candidates":
                    if candidates_egraph is not None:
                        used_egraph = candidates_egraph
                    else:
                        used_egraph = egraph.extract_to_postcondition(all_candidates=True)
                elif post_type == "representative":
                    used_egraph = get_representative_egraph(egraph, visualize)
                elif post_type == "yis":
                    used_egraph = get_yis_egraph(egraph, visualize)
                else:
                    raise ValueError(f"Invalid post_type: {post_type}")
            if output_path is not None:
                with open(output_path, "w") as f:
                    s = used_egraph.extract_to_sexpr_str()
                    logger.info("Post conditions:")
                    logger.info(s)
                    f.write(s)
                    LOGGER.info(f"Post conditions written into {output_path}")
            logger.print("Succeeded in finding post conditions.")
            logger.print(f"Extraction done in {datetime.now() - begin}")
        else:
            logger.error(f"{BRED}Error occurred in saturation process.{RST}")
            logger.error(saturation_process.communicate()[0].decode("utf-8"))
            raise EggError(f"Saturation failed, please check {output_log_path}")

    def check_result(self, dirname: str, raise_exception=True):
        output_log_path = self.get_output_log_path(dirname)
        with open(output_log_path, "r") as f:
            output_log = f.read()
        if (
            "Succeeded in finding post conditions." in output_log
            or "Passed equivalence checking." in output_log
            or "Done self provable checking." in output_log
        ):
            return True
        elif "Cannot find" in output_log:
            if raise_exception:
                raise CannotFindPostconditions(f"{BRED}Failed to find postconditions.{RST}\nPlease check {output_log_path}")
            else:
                return False
        elif "Failed equivalence checking" in output_log:
            if raise_exception:
                raise FailedImplyingEquivalence(f"{BRED}User expectation violated.{RST}\nPlease check {output_log_path}")
            else:
                return False
        else:
            raise EggError(f"{BRED}Unknown egg error.{RST}\nPlease check {output_log_path}")

    def download(self, dirname: str, postcondition=True) -> str:
        if self.use_local_directory:
            LOGGER.info(f"{BGREEN}No need to download: {dirname} ...{RST}")
        else:
            src_files = ["*.svg", "*.json", "*.log"]
            if postcondition:
                src_files = ["postcondition.sexpr"] + src_files
            src_paths = [osp.join(self.egg_data_dirname, f) for f in src_files]
            cmd = " ; ".join(f"cp -r {src_path} {dirname}" for src_path in src_paths)
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
            LOGGER.info(f"{BGREEN}Downloading postcondition to {dirname}...{RST}")
            p.wait()
            if p.returncode != 0:
                out, _ = p.communicate()
                LOGGER.info(out.decode())
                raise RuntimeError(f"Failed to download (returncode={p.returncode}).")
        if postcondition:
            os.system(f"mv {dirname}/postcondition.sexpr {dirname}/raw_postcondition.sexpr")

    def check_self_provable(self, args) -> list[int]:
        eclass_str: str = args[0]
        econd_name_shape_mappings: dict[str, ShapeLike] = args[1]
        if eclass_str.strip(" \n\r\t") == "":
            return []
        used_dirname = tempfile.mkdtemp()
        with open(osp.join(used_dirname, "precondition.scalar.smtlib"), "w") as f:
            f.write(self.all_scalar_econd_str)
        with open(osp.join(used_dirname, "impl_checklist.sexpr"), "w") as f:
            f.write(eclass_str)
        mapping = {n: shape.to_egg_str() for n, shape in econd_name_shape_mappings.items()}
        with open(osp.join(used_dirname, "shapes.json"), "w") as f:
            json.dump(mapping, f, indent=4)
        assert self.tmux == False, "Cannot use tmux mode for self provable checking."
        self.run(used_dirname, tmux_window_id=None, mode="self_provable", silent=True)
        self.check_result(used_dirname)
        provable_ids = open(osp.join(used_dirname, "result_ids.txt")).read()
        provable_ids = provable_ids.strip("\n").split(",")
        provable_ids = [int(r) for r in provable_ids]
        shutil.rmtree(used_dirname)
        return provable_ids

    def copy_pre_as_post(self, dirname: str):
        # Pass through the preconditions and computations as postconditions.
        post_str = ""
        for filename in ["precondition.sexpr"]:
            post_str += open(osp.join(dirname, filename)).read().strip("\n")
            post_str += "\n"
        with open(osp.join(dirname, "postcondition.sexpr"), "w") as f:
            f.write(post_str.strip("\n"))

    def get_postcondition(self, dirname: str, raw=False) -> str:
        if raw:
            dest_path = osp.join(dirname, "raw_postcondition.sexpr")
        else:
            dest_path = osp.join(dirname, "postcondition.sexpr")
        with open(dest_path, "r") as f:
            postcondtition_str = f.read()

        return postcondtition_str
