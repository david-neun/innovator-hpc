# =====================================================================
#  CSC/DSE 489/589 --- Project 0 receipt
#
#  Prints a receipt proving (a) you ran this, and (b) it ran on a
#  compute node under the scheduler.  Submit the slurm-<jobid>.out file
#  this produces, UNEDITED.  Do not retype it.
# =====================================================================

using Printf, LinearAlgebra, Dates

env(k) = get(ENV, k, "<<MISSING>>")
row(k, v) = @printf("  %-22s %s\n", k, v)

println("="^66)
println("  CSC/DSE 489/589 --- Project 0 receipt")
println("="^66)

row("user",            env("USER"))
row("timestamp (UTC)", Dates.format(now(UTC), "yyyy-mm-dd HH:MM:SS"))

println("\n-- scheduler ------------------------------------------------------")
for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION",
          "SLURM_JOB_NODELIST", "SLURM_SUBMIT_HOST", "SLURM_JOB_ACCOUNT",
          "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")
    row(lowercase(k)[7:end], env(k))
end

println("\n-- machine --------------------------------------------------------")
row("hostname",      gethostname())
row("cpu model",     Sys.cpu_info()[1].model)
row("logical cores", Sys.CPU_THREADS)
row("total memory",  @sprintf("%.1f GiB", Sys.total_memory() / 2^30))
row("julia version", VERSION)
row("julia threads", Threads.nthreads())
row("BLAS threads",  BLAS.get_num_threads())

println("\n-- warm-up --------------------------------------------------------")
n = 1500
A, B = rand(n, n), rand(n, n)
C = A * B                                   # compile + warm the caches
t = @elapsed A * B
row("matmul size",  "$n x $n")
row("elapsed",      @sprintf("%.3f s", t))
row("GFLOP/s",      @sprintf("%.1f", 2 * n^3 / t / 1e9))
row("sanity check", isapprox(C[1,1], dot(A[1,:], B[:,1])) ? "ok" : "FAILED")

# --- the thread-count trap from S06, checked out loud -----------------
cpt = env("SLURM_CPUS_PER_TASK")
if cpt != "<<MISSING>>" && string(Threads.nthreads()) != cpt
    println("\n  NOTE: julia threads ($(Threads.nthreads())) does not match")
    println("        --cpus-per-task ($cpt).  Your batch script is not giving")
    println("        Julia the cores you reserved.  See S06; fix it and rerun.")
end

# --- the receipt ------------------------------------------------------
jid = env("SLURM_JOB_ID")
println("\n" * "="^66)
println("  RECEIPT   user=$(env("USER"))   job=$jid   node=$(gethostname())")
println("="^66)

if jid == "<<MISSING>>"
    println()
    @error """
    SLURM_JOB_ID is not set, so this did NOT run as a scheduled job.
    You are on the login node or your own machine.  Submit with:
        sbatch project0.sbatch
    """
    exit(1)
end
