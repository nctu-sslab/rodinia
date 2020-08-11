#!/usr/bin/env python3

import os
import subprocess
import threading
import time
import datetime
import sys
import pickle
import copy
from termcolor import colored, cprint

import nvprof_parser
import libtarget_parser
from dataTy import dataTy
from dataTy import resultTy
from dataTy import Output

# global config
class Config:
    dry_run = False
    verifying = False
    enalbeProfile = True
    runStdSize = False
    test_count = 3
    test_delay = 0.5
    _run_cmd = "./run".split()
    _verify_cmd = "./verify".split()
    _clean_cmd = "make clean".split()
    _make_cmd = "make".split()

# Indivial option
class Opt:
    def __init__(self):
        self.env = copy.deepcopy(os.environ)
        self.timeout = 1000 # (s) ??
        self.reg_run_cmd = Config._run_cmd
        self.verify_cmd = Config._verify_cmd
        self.make_cmd = Config._make_cmd
        self.clean_cmd = Config._clean_cmd

        self.timer_out = ".time_out"
        # %p is the pid
        self.nvprof_out_prefix = ".nvprof_out"
        self.nvprof_out = self.nvprof_out_prefix + "%p"

        self.timer_prefix = ("/usr/bin/time -o " + self.timer_out + " -f %e").split()
        self.nvprof_prefix = ("nvprof -u s --log-file " + self.nvprof_out + " --profile-child-processes").split()

        self.cuda = False

class Test:
    def __init__(self, name, path, opt):
        self.name = name  # config name
        self.root = path
        self.opt = opt
    def run (self, projs, result):
        print(self.name)

        # init result
        self.result = resultTy()
        for proj in projs:
            os.chdir(self.root)
            # Real run
            print("* {:<15}".format(proj), end=' ', flush=True)
            cprint('....', 'yellow', attrs=['blink'], end='', flush=True)
            output = self.runOnProj(proj)
            if output.hasError():
                cprint("\b\b\b\b{0}    ".format(output.getError()), 'red')
            else:
                cprint("\b\b\b\bPass   ", 'green')
            self.result[proj] = output

        result[self.name] = self.result

    def runOnProj(self, proj):
        output = Output()

        os.chdir(project_path.get(proj, proj))

        self.opt.run_cmd = self.opt.reg_run_cmd

        #  TODO test verify

        # make clean
        subprocess.run(self.opt.clean_cmd, capture_output=True)
        # make
        # TODO add dry run
        CP = subprocess.run(self.opt.make_cmd, capture_output=True, env=self.opt.env)
        if CP.returncode != 0 :
            output.CE = True
            return output

        # Run/verify w/ timeout
        # Profiling and Get data
        for i in range(Config.test_count):
            ret = self.runWithTimer(output)
            if output.hasError(ret):
                return output
            pass
        if Config.enalbeProfile == True:
            for i in range(Config.test_count):
                ret = self.runWithProfiler(output)
                if output.hasError(ret):
                    return output
                ret = self.runWithNvprof(output)
                if output.hasError(ret):
                    return output
        # make clean
        #subprocess.run(self.opt.clean_cmd, capture_output=True)
        return output

    def runWithTimer(self, output):
        time.sleep(Config.test_delay)
        time_cmd = self.opt.timer_prefix + self.opt.run_cmd
        if Config.dry_run:
            print(time_cmd)
            return 0
        try:
            CP = subprocess.run(time_cmd, capture_output=True, timeout=self.opt.timeout, env=self.opt.env)
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1
        ret, timing = self.checkRet(CP, output, True)
        if ret != 0:
            return -1
        output.times.append(timing)
        return 0
    def runWithNvprof(self, output):
        # Run with export TMPDIR=/dev/shm to prevent tmp file full your /tmp
        nv_env = copy.deepcopy(self.opt.env)
        nv_env["TMPDIR"] = "/dev/shm"
        time.sleep(Config.test_delay)
        cmd = self.opt.nvprof_prefix + self.opt.run_cmd
        if Config.dry_run:
            print(cmd)
            return 0
        try:
            CP = subprocess.run(cmd, capture_output=True, timeout=self.opt.timeout, env=nv_env)
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1
        ret = self.checkRet(CP, output, False)
        if ret != 0:
            return -1
        # Process output in multiple self.opt.nvprof_out
        out_files = [ f for f in os.listdir(os.getcwd()) if self.opt.nvprof_out_prefix in f]
        for f in out_files:
            with open(f, 'r') as out:
                nvprof_result = out.read()
                ret = nvprof_parser.parse(output, nvprof_result)
                if ret == 0:
                    for f in out_files:
                        os.remove(f)
                    return 0
        for f in out_files:
            os.remove(f)
        return -1

    def runWithProfiler(self, output):
        time.sleep(Config.test_delay)
        prof_cmd = self.opt.timer_prefix + self.opt.run_cmd
        env = copy.deepcopy(self.opt.env)
        env["PERF"] = "1"
        if Config.dry_run:
            print(prof_cmd)
            return 0
        try:
            CP = subprocess.run(prof_cmd, capture_output=True, timeout=self.opt.timeout, env=env)
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1
        ret, timing = self.checkRet(CP, output, True)
        if ret != 0:
            return -1
        output.prof_times.append(timing)
        # Process output
        libtarget_parser.parse(output, CP.stderr.decode("utf-8"))
        return 0
    def checkRet(self, CP, output, getTime=False):
        # Store output
        output.stdouts += CP.stdout.decode("utf-8")
        output.stderrs += CP.stderr.decode("utf-8")
        if CP.returncode != 0 :
            output.RE = True
            #print("Run cmd: {0}".format(CP.args))
            #print(CP.returncode)
            #print(CP.stdout.decode("utf-8"))
            #print(CP.stderr.decode("utf-8"))
            return -1, -1
        if getTime:
            with open(self.opt.timer_out, 'r') as out:
                timing = float(out.read())
                os.remove(self.opt.timer_out)
                return 0, timing
            return -1, 0
        return 0

# RUN with CPU #######################################################
#FIXME comgare with CPU with O0, some benchmark are easy to opt??
def run_cpu1():
    opt = Opt()
    opt.env["OMP_NUM_THREADS"] = "1"
    return Test("omp-cpu1", os.path.join(rodinia_root, "openmp"), opt)

def run_cpu2():
    opt = Opt()
    opt.env["OMP_NUM_THREADS"] = "2"
    return Test("omp-cpu2", os.path.join(rodinia_root, "openmp"), opt)
def run_cpu4():
    opt = Opt()
    opt.env["OMP_NUM_THREADS"] = "2"
    return Test("omp-cpu2", os.path.join(rodinia_root, "openmp"), opt)
def run_cpu8():
    opt = Opt()
    opt.env["OMP_NUM_THREADS"] = "8"
    return Test("omp-cpu8", os.path.join(rodinia_root, "openmp"), opt)
######################################################################

def run_omp():
    opt2 = Opt()
    opt2.env["OFFLOAD"] = "1"
    return Test("omp-offload", os.path.join(rodinia_root, "openmp"), opt2)

def run_dce():
    opt_dce = Opt()
    opt_dce.env["OFFLOAD"] = "1"
    # Compile DCE with DC
    opt_dce.env["DC"] = "1"
    return Test("omp-dce", os.path.join(rodinia_root, "openmp"), opt_dce)

def run_bulk():
    opt3 = Opt()
    opt3.env["OFFLOAD"] = "1"
    opt3.env["OMP_BULK"] = "1"
    return Test("omp-offload-bulk", os.path.join(rodinia_root, "openmp"), opt3)

def run_dce_bulk():
    opt3 = Opt()
    opt3.env["OFFLOAD"] = "1"
    opt3.env["OMP_BULK"] = "1"
    opt3.env["DC"] = "1"
    return Test("dce-bulk", os.path.join(rodinia_root, "openmp"), opt3)

def run_bulk_host_shadow():
    opt3 = Opt()
    opt3.env["OFFLOAD"] = "1"
    opt3.env["OMP_BULK"] = "1"
    opt3.env["DC"] = "1"
    opt3.env["OMP_HOSTSHADOW"] = "1"
    return Test("dce-bulk-host-shadow", os.path.join(rodinia_root, "openmp"), opt3)

def run_at():
    opt4 = Opt()
    opt4.env["OFFLOAD"] = "1"
    opt4.env["OMP_BULK"] = "1"
    opt4.env["OMP_AT"] = "1"
    return Test("omp-offload-at", os.path.join(rodinia_root, "openmp"), opt4)

def run_dce_at():
    opt4 = Opt()
    opt4.env["OFFLOAD"] = "1"
    opt4.env["OMP_BULK"] = "1"
    opt4.env["OMP_AT"] = "1"
    opt4.env["DC"] = "1"
    return Test("dce-at", os.path.join(rodinia_root, "openmp"), opt4)

def run_maskat():
    opt = Opt()
    opt.env["OFFLOAD"] = "1"
    opt.env["OMP_MASK"] = "1"
    #opt.env["DC"] = "1"
    return Test("omp-mask-at", os.path.join(rodinia_root, "openmp"), opt)

def run_naive_at():
    opt = Opt()
    opt.env["OFFLOAD"] = "1"
    opt.env["OMP_BULK"] = "1"
    opt.env["OMP_NAIVE_AT"] = "1"
    opt.env["OMP_AT"] = "1"
    return Test("omp-offload-at-naive", os.path.join(rodinia_root, "openmp"), opt)

def run_cuda():
    opt5 = Opt()
    opt5.cuda = True
    return Test("cuda", os.path.join(rodinia_root, "cuda"), opt5)

# Change to refactor
def run_refactor():
    opt = Opt()
    opt.env["POLY1D"] = "1"
    opt.env["OFFLOAD"] = "1"
    opt.env["RUN_1D"] = "1"
    # FIXME change naming
    return Test("refactor", os.path.join(rodinia_root, "openmp"), opt)

def Setup():
    Tests = []

    poly_projs = ["2mm", "3mm", "atax", "bicg", "doitgen", "gemm", "gemver", "correlation", "covariance", "fdtd-apml", "convolution-2d", "reg_detect"]
    rodinia_projs = ["backprop", "kmeans", "myocyte", "pathfinder", "streamcluster"]
    rodinia_projs_no_atomic = ["backprop", "myocyte", "pathfinder"]

    ###################### Run mode options ###########################
    #Config.dry_run = True
    #Config.verifying = 1
    #Config.enalbeProfile = False
    #Config.runStdSize = True
    ###################################################################

    # Testing projects
    projects = poly_projs
    #projects = ["2mm"]
    #projects = ["backprop", "myocyte"]
    #projects = ["pathfinder"]

    #projects = rodinia_projs

    # Final result
    Config.test_count = 2
    #Tests.append(run_cuda)
    #Tests.append(run_bulk)
    #Tests.append(run_at)
    #Tests.append(run_dce)
    #Tests.append(run_bulk_host_shadow)

    Tests.append(run_omp)
    Tests.append(run_dce_bulk)
    Tests.append(run_dce_at)
    Tests.append(run_maskat)
    Tests.append(run_refactor)

    #Tests.append(run_cpu1)
    #Tests.append(run_cpu2)
    #Tests.append(run_cpu4)
    #Tests.append(run_cpu8)

    #Tests.clear()
    #Tests.append(run_maskat)

    return Tests , projects

def Pickle(Result):
    # save result to pickle
    os.chdir(script_dir)
    now = datetime.datetime.now()
    timestamp = now.strftime("%m_%d_%H_%M")
    pickle_file = "./results/result_" + timestamp + ".p"
    with open(pickle_file, "wb") as f:
        pickle.dump(Result, f)
    # save as last result
    pickle_file = "./results/result.p"
    with open(pickle_file, "wb") as f:
        pickle.dump(Result, f)

project_path = {}
polybench_path = "../../PolyBench-ACC/OpenMP/"
project_path["2mm"] = polybench_path + "linear-algebra/kernels/2mm/"
project_path["atax"] = polybench_path + "linear-algebra/kernels/atax/"
project_path["3mm"] = polybench_path + "linear-algebra/kernels/3mm/"
project_path["bicg"] = polybench_path + "linear-algebra/kernels/bicg/"
project_path["doitgen"] = polybench_path + "linear-algebra/kernels/doitgen/"
project_path["gemm"] = polybench_path + "linear-algebra/kernels/gemm/"
project_path["gemver"] = polybench_path + "linear-algebra/kernels/gemver/"
project_path["correlation"] = polybench_path + "datamining/correlation/"
project_path["covariance"] = polybench_path + "datamining/covariance/"
project_path["fdtd-apml"] = polybench_path + "stencils/fdtd-apml/"
project_path["convolution-2d"] = polybench_path + "stencils/convolution-2d/"
project_path["reg_detect"] = polybench_path + "medley/reg_detect/"

cuda_project_path = {}
cuda_polybench_path = "../../PolyBench-ACC/CUDA/"
cuda_project_path["2mm"] = cuda_polybench_path + "linear-algebra/kernels/2mm/"
cuda_project_path["atax"] = cuda_polybench_path + "linear-algebra/kernels/atax/"
cuda_project_path["3mm"] = cuda_polybench_path + "linear-algebra/kernels/3mm/"
cuda_project_path["bicg"] = cuda_polybench_path + "linear-algebra/kernels/bicg/"
cuda_project_path["doitgen"] = cuda_polybench_path + "linear-algebra/kernels/doitgen/"
cuda_project_path["gemm"] = cuda_polybench_path + "linear-algebra/kernels/gemm/"
cuda_project_path["gemver"] = cuda_polybench_path + "linear-algebra/kernels/gemver/"
cuda_project_path["correlation"] = cuda_polybench_path + "datamining/correlation/"
cuda_project_path["covariance"] = cuda_polybench_path + "datamining/covariance/"
cuda_project_path["fdtd-apml"] = cuda_polybench_path + "stencils/fdtd-apml/"
cuda_project_path["convolution-2d"] = cuda_polybench_path + "stencils/convolution-2d/"
cuda_project_path["reg_detect"] = cuda_polybench_path + "medley/reg_detect/"

Result = resultTy()
script_dir = os.path.dirname(os.path.realpath(__file__))
rodinia_root = os.path.dirname(script_dir)

def main():
    # Moving
    os.chdir(rodinia_root)

    TestGens, projects = Setup()
    if Config.verifying == True:
        Config.enalbeProfile = False
        Config.test_count = 1
        Config._run_cmd = Config._verify_cmd
        # For polybench
        os.environ["RUN_DUMP"] = "1"
        os.environ["RUN_MINI"] = "1"
    elif Config.runStdSize == False:
        os.environ["RUN_LARGE"] = "1"
        #os.environ["RUN_EXTRALARGE"] = "1"


    # print info
    if Config.dry_run == True:
        print("Dry-run, no data produced")
    if Config.enalbeProfile == True:
        cprint('Profiling', 'red', end='')
        print(" enabled")
    if Config.verifying == True:
        print("Start in ", end = '')
        cprint("verify", 'red', end='')
        print(" mode");
    if Config.runStdSize == True:
        cprint("Run Standard Size Data", 'blue')

    print("Start running duplicate test x{0}".format(Config.test_count))
    configs = []
    for TG in TestGens:
        configs.append(TG().name)
    cprint("Projects: ", 'blue', end='')
    print("{0}".format(', '.join(projects)))
    cprint("Configs: ", 'blue', end='')
    print("{0}".format(', '.join(configs)))

    for TG in TestGens:
        test = TG()
        test.run(projects, Result)
    Pickle(Result)

if __name__ == "__main__":
    try:
        main()
    except:
        print("Exception occurred")
        Pickle(Result)
        raise
