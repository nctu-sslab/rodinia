#!/usr/bin/env python3
import sys
import statistics

class dataTy:
    Name = ""
    Count = 0
    Value = 0
    min = 0
    max = 0
    avg = 0
    stdev = 0
    def __init__(self, name = "", count: int = 0, val : float = 0.0, _avg = 0, _min = 0, _max = 0):
        self.Name = name
        if not isinstance(count, int):
            print("Init of {0} dataTy.count is not int".format(name))
        if not isinstance(val, float):
            print("Init of {0} dataTy.val is not float".format(name))
        self.Count = count
        self.Value = val
        self.max = _max
        self.min = _min
        self.avg = _avg

# Per config, per proj
class Output:
    def __init__(self):
        self.CE = False
        self.RE = False
        self.TL = False
        self.Failed = False
        self.Error = False
        # output
        self.stdouts=""
        self.stderrs=""
        # list of execution time
        self.times = []
        self.time = 0 # avg
        # list of profiling result(dataTy for omp) dict
        self.nvprof_datas = []
        self.nvprof_times = []
        self.prof_datas = []
        self.prof_times = []
        # Finalized profiling result
        self.prof_data = {}
        #self.nvprof_data = {}
        self.prof_time = 0
        
        # Standard Deviation
        self.stdev_time = 0
        self.stdev_prof_time = 0
        # stderr of profiling mode
        # TODO
        self.logs = []
    def hasError(self, ret = 0):
        if ret != 0 or self.CE or self.RE or self.TL or self.TL or self.Failed:
            self.Error = True
            return True
        return False
    def getError(self):
        if self.CE:
            return "CE"
        if self.RE:
            return "RE"
        if self.TL:
            return "TL"
        if self.Failed:
            return "WA"
        return ""
    def getErrorOrAvgTime(self):
        if self.hasError() == True:
            return self.getError()
        return sum(self.times)/ len(self.times)

class resultTy(dict):
    pickle_path = ""
    
def getAvg(flist):
    if len(flist) < 1:
        return 0.0
    n = len(flist)
    sum = 0
    for s in flist:
        sum += s
    return sum / n

def getStdev(num_list):
    if len(num_list) < 2:
        return 0.0
    return statistics.stdev(num_list)
    
class ResultHelper:
    # get Projs of first config
    def getProjs(result):
        if len(result) < 1:
            print("No result")
            return []
        projs = {}
        for config in result:
            output = result[config]
            for p in output:
                projs[p] = 0
        ret = []
        for proj in projs:
            ret.append(proj)
        return ret
    def getConfigs(result):
        if len(result) < 1:
            print("No result")
            return []
        ret = []
        for config in result:
            ret.append(config)
        return ret

    def preprocessing(result):
        # show if result has error
        for config in result:
            for proj in result[config]:
                output = result[config][proj]
                if output.hasError() == True:
                    print(config + "-" + proj + " has error: " + output.getError())
        # Avg times
        for config in result:
            for proj in result[config]:
                output = result[config][proj]
                output.time = getAvg(output.times)
                output.stdev_time = getStdev(output.times)
        # Avg profiling result
        for config in result:
            for proj in result[config]:
                output = result[config][proj]
                # avg time
                output.prof_time = getAvg(output.prof_times)
                output.stdev_prof_time = getStdev(output.prof_times)
                
                important_metric = ["Runtime", "Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr"]

                if len(output.prof_datas) == 0:
                    # Fill 0 to important profile data
                    pdata = output.prof_data
                    for m in important_metric:
                        pdata[m] = dataTy(m, 1, 0.0)
                    continue
                    
                # avg data
                names = list(output.prof_datas[0].keys())
                for name in names:
                    the_sum = 0
                    for pdata in output.prof_datas:
                        the_sum += pdata.get(name,dataTy()).Value
                    val = the_sum / len(output.prof_datas)
                    count = output.prof_datas[0][name].Count
                    output.prof_data[name] = dataTy(name, count, val)
                    
                # Fill 0 to un-profiled important metrics
                for m in important_metric:
                    pdata = output.prof_data
                    if pdata.get(m) == None:
                        pdata[m] = dataTy(m, 1, 0.0)
                    
        # Do the nvprof data
        for config in result:
            for proj in result[config]:
                output = result[config][proj]
                if len(output.nvprof_datas) == 0:
                    continue
                nvprof_entries = list(output.nvprof_datas[0].keys())
                # Do the kernel first
                kernel_sum = 0
                kernel_count = 0
                # kernel time
                for name in nvprof_entries:
                    if name[:7] == "kernel-":
                        for pdata in output.nvprof_datas:
                            kernel_sum += pdata.get(name,dataTy()).Value
                        kernel_count += output.nvprof_datas[0][name].Count
                output.prof_data["GPU-kernel"] = dataTy("GPU-kernel", kernel_count, kernel_sum/len(output.nvprof_datas))
                for name in nvprof_entries:
                    if name[:7] == "kernel-":
                        continue
                    the_sum = 0
                    for pdata in output.nvprof_datas:
                        the_sum += pdata.get(name,dataTy()).Value
                    val = the_sum / len(output.nvprof_datas)
                    count = output.nvprof_datas[0][name].Count
                    output.prof_data[name] = dataTy(name, count, val)
        # Substract runtime with others
        for config in result:
            for proj in result[config]:
                output = result[config][proj]
                pdata = output.prof_data
                member = ["Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr"]
                get = pdata.get("Runtime")
                if get == None:
                    #pdata["Runtime"] = dataTy("Runtime", 1, 0.0) 
                    continue

                sumup = get.Value
                for m in member:
                    sumup -= pdata[m].Value
                if sumup < 0:
                    sumup = 0
                # Store OMP Runtime in new attr
                pdata["OMPRuntime"] = dataTy("OMPRuntime", 1, sumup)
        # Gen other
        for config in result:
            for proj in result[config]:
                output = result[config][proj]
                pdata = output.prof_data
                member = ["Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr", "OMPRuntime"]
                sumup = output.prof_time
                for m in member:
                    sumup -= pdata[m].Value
                if sumup < 0:
                    sumup = 0
                d = dataTy("Other", 1, sumup)
                pdata["Other"] = d
        # Avg times and store into pdata
        for config in result:
            for proj in result[config]:
                output = result[config][proj]
                pdata = output.prof_data
                AvgTime = getAvg(output.times)
                pdata["Times"] = dataTy("Times", 1, AvgTime)
        # sort the config??
    def isInvalid(result):
        projs = ResultHelper.getProjs(result)
        configs = ResultHelper.getConfigs(result)
        proj_count = len(projs)
        config_count = len(configs)
        if config_count == 0:
            print("result is invalid")
            return True
        if proj_count == 0:
            print("result is invalid")
            return True
        return False

    def getNormFactors(outputs_of_the_config, metrics, norm = True):
        outputs = outputs_of_the_config
        factors = {p: 1 for p in outputs}
        # Norm to first
        if norm == True:
            for p in outputs:
                sum = 0
                for m in metrics:
                    sum += outputs[p].prof_data[m].Value
                if sum == 0:
                    print()
                    factors[p] = 1
                else:
                    factors[p] = 100/sum
        return factors
