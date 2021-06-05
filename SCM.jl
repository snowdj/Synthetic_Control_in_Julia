using CSV, HTTP, DataFrames, Statistics, Plots, Dates

# Get Data 
link = "https://github.com/azev77/Synthetic_Control_in_Julia/raw/main/HP_q.csv"
r = HTTP.get(link)
df = CSV.read(r.body, DataFrame, missingstring="NA")

# Clean Data 
df = df[(df.sizerank .<= 250), :]              #keep sizerank <=100     9389/41  =229
df = df[(df.loc .!=  "Bancroft_and_Area"), :]  #Bad fitting pre         9348/41  =228
df = df[(df.loc .!=  "Australia"), :]          #Polluted                9266/41  =226 
df = df[(df.loc .!=  "average8capital"), :]    #Aus Polluted            9225/41  =225

unique(df.loc)                                 #
df.loc = df.loc .* "_" .* df.id 
unique(df.loc)                                 #

u_treated = ["Greater_Vancouver_Canada"]
u_control = df[(df.i_treated .==0), "loc"] |> unique #


#Quarterly: last quarter pre treatment. 
#Vertical line Pre 1st tax 
function _T0(x)
    if x in(["Greater_Vancouver_Canada"])     
        rT0 = 26 #df.Date[26]  "1-Jun-16"   
    else
        rT0 = nothing                                                          
    end
    return rT0    
end
[println(_T0(u_treated[i])) for i in 1:length(u_treated)]

#Vertical lines at ALL taxes 
function _TA(x)
    if x in(["Greater_Vancouver_Canada"])   
        rT0 = [26.0; 33.0]
    else
        rT0 = nothing  
    end
    return rT0    
end
[println(_TA(u_treated[i])) for i in 1:length(u_treated)]

#
outcome = "HPGyy"  #outcome = "HPGqq"
d    = filter(:loc => in(u_control), df)    #remove treated cities from df 
du, dt = d.loc, d.Date
unit, time = unique(du), unique(dt) # 199 locs, 41 quarters = 8159 obs
T = length(time)                    # 41 q 
N_id = size(unit,1) + 1             # 181
X = d[:, outcome]
X = reshape(X, T, N_id - 1)
X = [X 1.0:Float64(T) (1.0:Float64(T)).^2.0] # Add Linear & Quadratic Time trends.

#Plot HPG & taxes for each treated city. 
for i in 1:length(u_treated)
    #i = 2
    println("unit = ", u_treated[i])
    T0 =  _T0(u_treated[i])
    TAX = _TA(u_treated[i])
    y = df[(df.loc .== u_treated[i]), outcome]  # treated unit 
    time = 1:T
    #"Plot Y"
    plt_Y = plot(legend = :topleft)
    plot!([T0], seriestype = :vline, lab="Tax", color="black")
    TAX != nothing ? plot!(TAX, seriestype = :vline, lab="Tax", color="black") : nothing  
    plot!([0.0],  seriestype = :hline, lab="", color="red")  
    plot!(time, y, lab =u_treated[i], color="green")
    plt_Y |> display
    #savefig(plt_Y, f_output*"Y_"*u_treated[i]*".png")
end

using GLMNet
scpr(m,x)  = GLMNet.predict(m, x)

#Tune HP for each treated city. 
for i in 1:length(u_treated)
    #i = 1
    println("unit = ", u_treated[i])    
    ##########################################################################
    #                         MAKE DATA 
    ##########################################################################    
    T0 =  _T0(u_treated[i])
    T1 = (T - T0)
    t_pre = 1:T0
    t_post = (T0 + 1):T
    #
    y = df[(df.loc .== u_treated[i]), outcome]  # treated unit 
    X_pre,  y_pre    = X[t_pre,:],   y[t_pre]
    X_post,  y_post  = X[t_post,:],  y[t_post]
    ##########################################################################
    #                         Tune HP  
    ##########################################################################    
    #CV1: tune ALPHA
    grid_a = [0.0 : 0.1 : 1.0;]
    HPscore  = []
    for α in grid_a
        println("alpha = ", α)
        #
        ty = y  # treated unit 
        tX = X    # control group
        #
        ty_pre = ty[t_pre]
        tX_pre = tX[t_pre,:]
        #
        m  = glmnetcv(tX_pre, ty_pre, nlambda=100, alpha = α, nfolds =10)#
        push!(HPscore, minimum(m.meanloss))
    end 
    opt_α = grid_a[argmin(HPscore[:])]
    #CV2: tune LAMBDA 
    m  = glmnetcv(X_pre, y_pre, nlambda=1000, alpha = opt_α, nfolds =10)
    opt_λ = m.lambda[argmin(m.meanloss)]
    #
    #
    #
    scfit(x,y) = glmnet(x, y, lambda=[opt_λ], alpha = opt_α)
    ##########################################################################
    #                         Fit Treated unit.
    ##########################################################################    
    m_L2 = scfit(X_pre, y_pre)
    ŷ_L2 = scpr(m_L2, X)
    ATET_L2  = y - ŷ_L2
    #"LOOCV for treated unit for Pre-treatment years"
    # sc = []
    # for tt in t_pre
    #     ty = y_pre[1:end .!= tt]
    #     tX = X_pre[1:end .!= tt,:]
    #     m = scfit(tX, ty)         #glmnetcv(tX, ty, nlambda=10_000, alpha = 0.0)
    #     ŷ = scpr(m, X)            #GLMNet.predict(m, X)
    #     push!(sc, abs(ŷ[tt] - y[tt]))
    # end 
    # sc_L2 = sc
    # mean(sc_L2)  # MAE 
    ##########################################################################
    #                         Fit Placebo units & Imbens unit se_i.
    ##########################################################################       
    ATET = []
    std_err_i = zeros(N_id - 1, T1)
    for i in 1:(N_id - 1) #size(X,2)
        println("unit = ", i )   #u_control[i]
        ty = X[:,i]              #placebo treated unit 
        tX = X[:,1:end .!= i]    #placebo control group
        #
        ty_pre = ty[t_pre]
        tX_pre = tX[t_pre,:]
        #
        m  = scfit(tX_pre, ty_pre) #
        tŷ = scpr(m, tX)  # 
        #
        push!(ATET, ty - tŷ)
        std_err_i[i,:] = (ty[t_post] - tŷ[t_post]).^2
    end     
    se_i = [std_err_i[:,tt] |> mean |> sqrt for tt in 1:T1 ]
    "se_i(t): avg oos RMSE across all controls in yr t"
    ##########################################################################
    #                         Estimate CI/SE: 
    ##########################################################################
    "Jackknife+ CI algorithm over time"
    res, yp = [], []
    for tt in t_pre #1:T0
        ty = y_pre[t_pre .!= tt]    #leave out tt
        tX = X_pre[t_pre .!= tt,:]  #Control leave out tt
        #
        m = scfit(tX, ty)          #glmnetcv(tX, ty, nlambda=10_000, alpha = 0.0) #
        ŷ = scpr(m, X)             #GLMNet.predict(m, X)
        #
        push!(res, y[tt] - ŷ[tt])   #res = y[tt] - ŷ[tt]
        push!(yp,  ŷ[t_post])       #yp  = ŷ[t_post] 
    end 
    xxxp = [ yp[tt] .+ abs(res[tt]) for tt in 1:T0 ]
    xxxp = hcat(xxxp...) #T1 rows, T0 cols. Each row = ECDF for SE
    xxxm = [ yp[tt] .- abs(res[tt]) for tt in 1:T0 ]
    xxxm = hcat(xxxm...) #T1 rows, T0 cols. Each row = ECDF for SE  
    alpha = 0.05
    LBJp = [quantile(xxxm[tt,:],       alpha/2.0)  for tt in 1:T1]
    UBJp = [quantile(xxxp[tt,:], 1.0 - alpha/2.0)  for tt in 1:T1]
    ATET_LBJp = [zeros(T0); y[t_post] -  LBJp] 
    ATET_UBJp = [zeros(T0); y[t_post] -  UBJp]
    ##########################################################################
    #                         Make Plots: 
    ##########################################################################        
    time = 1:T
    #"Plot Y(1) vs Ŷ(0)"
    TAX = _TA(u_treated[i])
    #
    plt_Y0 = plot(legend = :topleft)
    plot!(time, y, lab =u_treated[i])
    plot!([T0], seriestype = :vline, lab="", color="red")
    #plot!(TAX, seriestype = :vline, lab="", color="red")
    TAX != nothing ? plot!(TAX, seriestype = :vline, lab="", color="red") : nothing  
    plot!([0.0],  seriestype = :hline, lab="", color="red")  # |> display
    plot!(time, ŷ_L2, lab="synthetic") 
    plt_Y0 |> display
    #savefig(plt_Y0, f_output*"Y0_"*u_treated[i]*".png")
    #
    "Plot: ATET & Placebos"
    plt_ATT = plot(legend = :bottomleft)
    #plot!([T0], seriestype = :vline, lab="", color="red")
    plot!([T0], seriestype = :vline, lab="", color="red")
    #plot!(TAX, seriestype = :vline, lab="", color="red")
    TAX != nothing ? plot!(TAX, seriestype = :vline, lab="", color="red") : nothing  
    plot!([0.0],  seriestype = :hline, color="red", lab="")
    plot!(time, ATET, lab="")
    plot!(time, mean(hcat(ATET...), dims = 2), color="green", linewidth = 2, lab="mean donor")
    plot!(time, ATET_L2, color="red", linewidth = 2, lab =u_treated[i]) 
    plt_ATT |> display
    #savefig(plt_ATT, f_output*"ATT_"*u_treated[i]*".png")
    "Plot CI"
    plt_CI = plot(legend = :bottomleft)
    #plot!([T0], seriestype = :vline, lab="", color="red")
    plot!([T0], seriestype = :vline, lab="", color="red")
    #plot!(TAX, seriestype = :vline, lab="", color="red")
    TAX != nothing ? plot!(TAX, seriestype = :vline, lab="", color="red") : nothing  
    plot!([0.0],  seriestype = :hline, color="red", lab="")
    plot!(time, ATET_L2, color="red", linewidth = 2, lab =u_treated[i]*" ATT") 
    plot!(time, ATET_LBJp, color="green", linestyle = :dash, lab =u_treated[i]*" 95% CI")
    plot!(time, ATET_UBJp, color="green", linestyle = :dash, lab="")
    plt_CI |> display
    #savefig(plt_CI, f_output*"CI_"*u_treated[i]*".png")
end





