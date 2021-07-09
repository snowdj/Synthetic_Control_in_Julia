using CSV, HTTP, DataFrames, Statistics, Plots, Dates, Random

# Get Data 
link = "https://github.com/azev77/Synthetic_Control_in_Julia/raw/main/HP_q.csv"
r = HTTP.get(link)
df = CSV.read(r.body, DataFrame, missingstring="NA")

# Clean Data 
if 1==1
    df = df[(df.sizerank .<= 250), :]              #keep sizerank <=100     9389/41  =229
    #df = df[(df.loc .!=  "Bancroft_and_Area"), :]  #Bad fitting pre         9348/41  =228
    df = df[(df.loc .!=  "Australia"), :]          #Polluted                9266/41  =226 
    df = df[(df.loc .!=  "average8capital"), :]    #Aus Polluted            9225/41  =225
    df = df[(df.loc .!=  "SKorea"), :]      # drop Duplicate 

    unique(df.loc)                                 #
    df.loc = df.loc .* "_" .* df.id 
    unique(df.loc)                                 #

    u_treated = ["Greater_Vancouver_Canada"]   |> unique
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

end

#
outcome = "HPGyy"  
d    = filter(:loc => in(u_control), df)    #remove treated cities from df 
du, dt = d.loc, d.Date
unit, time = unique(du), unique(dt) # 345 locs, 41 quarters
time_ix = eachindex(time)
ttt = SubString.(time, 6, 9)        # String of years for plots... 
xt = (1:4:41, ttt[1:4:41])  #x ticks. Every 4Q. Only display year for Q1.

T = length(time)                    # 41 q 
N_id = size(unit,1) + 1             # 346
X = d[:, outcome]
X = reshape(X, T, N_id - 1)
X = [X 1.0:Float64(T) (1.0:Float64(T)).^2.0] # Add Linear & Quadratic Time trends.


#Plot HPG & taxes for the treated city. 
for i in 1:length(u_treated)
    #
    i = 1
    println("unit = ", u_treated[i])
    T0 =  _T0(u_treated[i])
    y   = df[(df.loc .== u_treated[i]), outcome]  # treated unit 
    x = [1.0:1.0:14.0;] #14 treated time periods 
    yc1 = [y[1:(T0+1)]; y[T0+1] .+ 2*log.(x .+ 1) ]
    yc2 = [y[1:(T0+1)]; y[T0+1] .- 22.5*log.(x .+ 1) ]
    #"Plot Y"
    plt_Y = plot(legend = :topleft, ylim = (-35,40))
    plot!([T0] .+1, seriestype = :vline, lab="Foreign Buyer Tax", color="red")
    plot!([0.0],  seriestype = :hline, lab="", color="red")  
    plot!(time_ix, y, xticks = xt, lab ="Y(1): Vancouver HPG observed", c=1)
    plot!(time_ix, yc1, xticks = xt, lab ="Y(0): Vancouver HPG counterfactual 1", c=4)
    plot!(time_ix, yc2, xticks = xt, lab ="Y(0): Vancouver HPG counterfactual 2", c=4, s=:dash)
    plt_Y |> display
end



using GLMNet
scpr(m,x)  = GLMNet.predict(m, x)

nf = 10;  # number folds for CV 
#nf = 26; # LOOCV 

#Tune HP for each treated city. 
for i in 1:length(u_treated)
    #
    i=1
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
    #CV1: tune ALPHA: 10 fold CV during pre-treatment 
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
        m  = glmnetcv(tX_pre, ty_pre, nlambda=100, alpha = α, nfolds =nf, rng=MersenneTwister(123))#
        push!(HPscore, minimum(m.meanloss))
    end 
    opt_α = grid_a[argmin(HPscore[:])]
    #CV2: tune LAMBDA: 10 fold CV during pre-treatment  
    m  = glmnetcv(X_pre, y_pre, nlambda=1000, alpha = opt_α, nfolds =nf, rng=MersenneTwister(123))
    opt_λ = m.lambda[argmin(m.meanloss)]
    #
    #m_w  = glmnetcv(X_pre, y_pre, nlambda=1000, alpha = opt_α, nfolds =10, rng=MersenneTwister(123))
    m_w  = glmnetcv(X_pre, y_pre, lambda=[opt_λ], alpha = opt_α, nfolds =nf, rng=MersenneTwister(123))
    #
    scfit(x,y) = glmnet(x, y, lambda=[opt_λ], alpha = opt_α)
    ##########################################################################
    #                         Fit Treated unit.
    ##########################################################################    
    m_L2 = scfit(X_pre, y_pre)
    ŷ_L2 = scpr(m_L2, X)
    ATET_L2  = y - ŷ_L2
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
    #                         Make Plots: 
    ##########################################################################        
    "Plot weights, θ, from Y=F(X, θ)"
    vn   = [u_control;"t1"; "t2"; "t0"]  # 348 element vec names of predictors
    coef = [ m_w.path.betas; m_w.path.a0[1]; ]
    nz = [coef[i,1] != 0.0 for i in 1:size(coef,1)] # non-zero coefs
    coef1 = coef[nz,:] # keep ≂̸ 0
    vn1   = vn[  nz,:] # keep ≂̸ 0
    sum(nz)     # non-zero predictors 
    sum(coef1)  #   == sum(coef)
    #
    p1 = bar(vn, coef, xrotation = 45, xticks =:all, lab="weight") 
    p2 = bar(vn1, coef1, xrotation = 45, xticks =:all, lab="weight") 
    plot(p1,p2,size=1.35 .* (600, 400), legend=:topleft)
  
    
    #
    "Plot Y(1) vs Ŷ(0)"
    plt_Y0 = plot(legend = :topleft)
    plot!([T0], seriestype = :vline, lab="Last quarter pre-treatment", color="red")
    plot!(time_ix, y, xticks=xt, c=1, lab="Y(1): Vancouver HPG observed")
    plot!(time_ix, ŷ_L2,         c=4, lab="Y(0): Synthetic Vancouver HPG")
    plot!([0.0],  seriestype = :hline, lab="", color="red")  # |> display
    

    #
    "Plot: ATET & Placebos"
    plt_V = plot(legend = :bottomleft)
    plot!([T0], seriestype = :vline, lab="Last quarter pre-treatment", color="red")
    plot!([0.0],  seriestype = :hline, color="red", lab="")
    plot!(time_ix, ATET_L2, xticks=xt, c=1, linewidth = 2, lab ="Vancouver ATT") 


    #
    "Plot: ATET & Placebos"
    plt_ATT = plot(legend = :bottomleft)
    plot!([T0], seriestype = :vline, lab="Last quarter pre-treatment", color="red")
    #
    plot!(time_ix, ATET[1], xticks=xt, color="gray", lab="Donor ATT")
    plot!(time_ix, ATET, color="gray", lab="")
    plot!(time_ix, mean(hcat(ATET...), dims = 2), color="green", linewidth = 2, lab="Mean Donor ATT")
    plot!([0.0],  seriestype = :hline, color="red", lab="")
    plot!(time_ix, ATET_L2, color=1, linewidth = 2, lab ="Vancouver ATT") 
end


plot(plt_Y0,plt_V, layout=(2,1),size=(600,400).*1.25)




using DataFrames, FixedEffectModels  #, BenchmarkTools
d = filter(:loc => in([u_control;u_treated]), df)    #remove treated cities from df 
# unique(d.loc)
# unique(d.loc[d.i_treated.==1])
dd = DataFrame(loc = d.loc, tim = d.Date, id = d.id,
                Y=d.HPGyy, tr=d.i_treated, t=repeat(1:41,346),);
dd.co = (dd.tr .==0)    
dd.post = Int.((dd.t  .>=27))            
dd.post_tr = Int.((dd.post .==1) .* (dd.tr .==1))     

dd.pre = Int.((dd.t  .<27))            
dd.t2 = (dd.t).^2

reg(dd, @formula(Y ~ 1 + tr + post +  post_tr ), Vcov.cluster(:loc))
#"post_tr     | -3.40421      0.2196   -15.5019    0.000  -3.83466  -2.97377"
"post_tr     | -3.43805   0.220487   -15.593    0.000  -3.87024  -3.00587"

reg(dd, @formula(Y ~ 1 + post_tr + fe(tim) +fe(loc)), Vcov.cluster(:loc))
"post_tr | -3.43805  0.211521 -16.2539    0.000  -3.85266  -3.02344"

reg(dd, @formula(Y ~ 1 + post_tr + fe(tim) +fe(loc) 
    #+ t&pre&fe(id)    # #+ t*pre*id NOT SAME??? 
    #
    #+ t&pre&fe(loc) + t2&pre&fe(loc)  #-20%
    + t&pre&fe(tr) + t2&pre&fe(tr)     #Same as ABOVE!
    ), Vcov.cluster(:loc))
"post_tr | -20.8762  0.538658 -38.7559    0.000   -21.932  -19.8203"


aa=unique(sort(dd.t))
for jj in 1:size(aa,1)
    col = "t" * string(jj)           #string(jj) # string(aa[jj])
    dd[!, col] = Int.( (dd.tr .==1) .* (dd.t .== aa[jj]) )
end
PT = map(x -> "t$x", [1:25; 27:41]) 
f = term(:Y) ~ sum(term.(PT)) + @formula(0~fe(loc) + fe(tim)
    #+ t&pre&fe(tr) + t2&pre&fe(tr) 
    #+ t&pre&fe(tr) + t2&pre&fe(tr) #+ t2&pre&fe(tr) 
    #+ t&fe(id) + t2&fe(id)
    #+ t&post&fe(loc) + t2&post&fe(loc)
    ).rhs
m_did = reg(dd, f)
coef_did = [m_did.coef[1:25]; 0.0;  m_did.coef[26:40]]
ci = confint(m_did; level = 0.95)
ci_did = [ci[1:25,:]; zeros(1,2);  ci[26:40,:]]

mean(m_did.coef[26:40]) #-23.250146422021242
mean(coef_did)

# ATT := Y(1) - Y(0)
# Y(1) - ATT = Y(0) 
y   = df[(df.loc .== u_treated[1]), outcome]
T0=26
Y0hat = y .- coef_did
p1=plot(legend=:bottomleft);
plot!([T0] , seriestype = :vline, lab="", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red");
plot!(coef_did, xticks=xt, lab="ATT")
plot!(ci_did, l=:dash, c=2 ,lab="95% CI")
#
p2=plot(legend=:bottomleft)
plot!([T0] , seriestype = :vline, lab="Last quarter pre-treatment", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red")
plot!(y, c=1, xticks=xt, lab="Y(1): Vancouver HPG Observed")
plot!(Y0hat, c=4, lab="Y(0): Synthetic Vancouver HPG")

#
plot(p2,p1, layout=(2,1),size=(600,400).*1.25)
plot(plt_Y0,plt_V, layout=(2,1),size=(600,400).*1.25)


# Hacked 
ix=24:41 # 24 is Q4-2015
T0
xt2 = (2:4:(41-24+1), ttt[25:4:41])
p1=plot(legend=:bottomleft);
plot!([T0-24+1] , seriestype = :vline, lab="", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red")  ;
plot!(coef_did[ix], xticks=xt2, lab="ATT")
plot!(ci_did[ix,:], l=:dash, c=2 ,lab="95% CI")
scatter!(ci_did[ix,:], l=:dash, c=2 ,lab="")

# Horizontal Regression AR(T0)
lag(dd.Y, 1)
f = @formula(Y ~ 1 + lag(Y, 1) )
reg(dd, f)

























#Int.((dd.t  .!=27) .* (dd.tr .==1))
#dd.pt1=Int.((dd.t.==1).*(dd.tr.==1))
if 1==1
    dd.pt1=Int.((dd.t.==1).*(dd.tr.==1))
    dd.pt2=Int.((dd.t.==2).*(dd.tr.==1))
    dd.pt3=Int.((dd.t.==3).*(dd.tr.==1))
    dd.pt4=Int.((dd.t.==4).*(dd.tr.==1))
    dd.pt5=Int.((dd.t.==5).*(dd.tr.==1))
    dd.pt6=Int.((dd.t.==6).*(dd.tr.==1))
    dd.pt7=Int.((dd.t.==7).*(dd.tr.==1))
    dd.pt8=Int.((dd.t.==8).*(dd.tr.==1))
    dd.pt9=Int.((dd.t.==9).*(dd.tr.==1))
    dd.pt10=Int.((dd.t.==10).*(dd.tr.==1))
    dd.pt11=Int.((dd.t.==11).*(dd.tr.==1))
    dd.pt12=Int.((dd.t.==12).*(dd.tr.==1))
    dd.pt13=Int.((dd.t.==13).*(dd.tr.==1))
    dd.pt14=Int.((dd.t.==14).*(dd.tr.==1))
    dd.pt15=Int.((dd.t.==15).*(dd.tr.==1))
    dd.pt16=Int.((dd.t.==16).*(dd.tr.==1))
    dd.pt17=Int.((dd.t.==17).*(dd.tr.==1))
    dd.pt18=Int.((dd.t.==18).*(dd.tr.==1))
    dd.pt19=Int.((dd.t.==19).*(dd.tr.==1))
    dd.pt20=Int.((dd.t.==20).*(dd.tr.==1))
    dd.pt21=Int.((dd.t.==21).*(dd.tr.==1))
    dd.pt22=Int.((dd.t.==22).*(dd.tr.==1))
    dd.pt23=Int.((dd.t.==23).*(dd.tr.==1))
    dd.pt24=Int.((dd.t.==24).*(dd.tr.==1))
    dd.pt25=Int.((dd.t.==25).*(dd.tr.==1))
    #dd.pt26=Int.((dd.t.==26).*(dd.tr.==1))
    dd.pt27=Int.((dd.t.==27).*(dd.tr.==1))
    dd.pt28=Int.((dd.t.==28).*(dd.tr.==1))
    dd.pt29=Int.((dd.t.==29).*(dd.tr.==1))
    dd.pt30=Int.((dd.t.==30).*(dd.tr.==1))
    dd.pt31=Int.((dd.t.==31).*(dd.tr.==1))
    dd.pt32=Int.((dd.t.==32).*(dd.tr.==1))
    dd.pt33=Int.((dd.t.==33).*(dd.tr.==1))
    dd.pt34=Int.((dd.t.==34).*(dd.tr.==1))
    dd.pt35=Int.((dd.t.==35).*(dd.tr.==1))
    dd.pt36=Int.((dd.t.==36).*(dd.tr.==1))
    dd.pt37=Int.((dd.t.==37).*(dd.tr.==1))
    dd.pt38=Int.((dd.t.==38).*(dd.tr.==1))
    dd.pt39=Int.((dd.t.==39).*(dd.tr.==1))
    dd.pt40=Int.((dd.t.==40).*(dd.tr.==1))
    dd.pt41=Int.((dd.t.==41).*(dd.tr.==1))
end

fdid = @formula(Y ~ fe(loc) + fe(tim)
+ pt1 + pt2 + pt3 + pt4 + pt5 
+ pt6 + pt7 + pt8 + pt9 + pt10 
+ pt11 + pt12 + pt13 + pt14 + pt15 
+ pt16 + pt17 + pt18 + pt19 + pt20 
+ pt21 + pt22 + pt23 + pt24 + pt25 #+ pt26 
       + pt27 + pt28 + pt29 + pt30 
+ pt31 + pt32 + pt33 + pt34 + pt35 
+ pt36 + pt37 + pt38 + pt39 + pt40 
+ pt41
)
#
#fieldnames(FixedEffectModel)
#m_did.coef
# m_did = reg(dd, fdid, Vcov.cluster(:loc,:t))
# m_did = reg(dd, fdid, Vcov.cluster(:loc))
# m_did = reg(dd, fdid, Vcov.cluster(:t))
# m_did = reg(dd, fdid, Vcov.robust())
m_did = reg(dd, fdid)
coef_did = [m_did.coef[1:25]; 0.0;  m_did.coef[26:40]]
ci = confint(m_did; level = 0.95)
ci_did = [ci[1:25,:]; zeros(1,2);  ci[26:40,:]]




# ATT := Y(1) - Y(0)
# Y(1) - ATT = Y(0) 
Y0hat = y .- coef_did
p1=plot(legend=:bottomleft);
plot!([T0] , seriestype = :vline, lab="", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red")  ;
plot!(coef_did, xticks=xt, lab="ATT")
plot!(ci_did, l=:dash, c=2 ,lab="95% CI")
#
p2=plot(legend=:bottomleft)
plot!([T0] , seriestype = :vline, lab="Last quarter pre-treatment", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red")
plot!(y, c=1, xticks=xt, lab="Y(1): Vancouver HPG Observed")
plot!(Y0hat, c=4, lab="Y(0): Synthetic Vancouver HPG")

#
plot(p2,p1, layout=(2,1),size=(600,400).*1.25)
plot(plt_Y0,plt_V, layout=(2,1),size=(600,400).*1.25)


# Hacked 
ix=24:41 # 24 is Q4-2015
T0
xt2 = (2:4:(41-24+1), ttt[25:4:41])
p1=plot(legend=:bottomleft);
plot!([T0-24+1] , seriestype = :vline, lab="", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red")  ;
plot!(coef_did[ix], xticks=xt2, lab="ATT")
plot!(ci_did[ix,:], l=:dash, c=2 ,lab="95% CI")
scatter!(ci_did[ix,:], l=:dash, c=2 ,lab="")


























#########################################################################
fdid = @formula(Y ~ fe(loc) + fe(tim)
+ pt1 + pt2 + pt3 + pt4 + pt5 
+ pt6 + pt7 + pt8 + pt9 + pt10 
+ pt11 + pt12 + pt13 + pt14 + pt15 
+ pt16 + pt17 + pt18 + pt19 + pt20 
+ pt21 + pt22 + pt23 + pt24 + pt25 #+ pt26 
       + pt27 + pt28 + pt29 + pt30 
+ pt31 + pt32 + pt33 + pt34 + pt35 
+ pt36 + pt37 + pt38 + pt39 + pt40 
+ pt41
# loc/tim #tr/t/id/co 
#+ t*id
+ t^2 *id
)
#
# Vcov.cluster(:loc,:t) #Vcov.robust()
m_did = reg(dd, fdid)
coef_did = [m_did.coef[1:25]; 0.0;  m_did.coef[26:40]]
ci = confint(m_did; level = 0.95)
ci_did = [ci[1:25,:]; zeros(1,2);  ci[26:40,:]]

Y0hat = y .- coef_did
p1=plot(legend=:bottomleft);
plot!([T0] , seriestype = :vline, lab="", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red")  ;
plot!(coef_did, xticks=xt, lab="ATT")
plot!(ci_did, l=:dash, c=2 ,lab="95% CI")
#
p2=plot(legend=:bottomleft)
plot!([T0] , seriestype = :vline, lab="Last quarter pre-treatment", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red")
plot!(y, c=1, xticks=xt, lab="Y(1): Vancouver HPG Observed")
plot!(Y0hat, c=4, lab="Y(0): Synthetic Vancouver HPG")


using InteractiveFixedEffectModels
fdid = @formula(Y ~ fe(loc) + fe(tim)
+ pt1 + pt2 + pt3 + pt4 + pt5 
+ pt6 + pt7 + pt8 + pt9 + pt10 
+ pt11 + pt12 + pt13 + pt14 + pt15 
+ pt16 + pt17 + pt18 + pt19 + pt20 
+ pt21 + pt22 + pt23 + pt24 + pt25 #+ pt26 
       + pt27 + pt28 + pt29 + pt30 
+ pt31 + pt32 + pt33 + pt34 + pt35 
+ pt36 + pt37 + pt38 + pt39 + pt40 
+ pt41
# loc/tim #tr/t/id/co 
#+ t*id
+ t^2 *id
+ ife(State, Year, 2)
)
m_did = regife(dd, @formula(Y ~ tr) )




###########################################################
























mean(dd.Y) # 2.88095 mean HPG in sample including Vancouver
f=@formula(Y ~ 1 )
m1 = reg(dd, f)
#(Intercept) |  2.8809
#pr = FixedEffectModels.predict(m1, dd)[dd.tr.==1,:]

plot();
plot!(y, lab="Y(1)");
plot!(m1.coef[1]*ones(41), lab="Y(0)")





mean(dd.Y) # 2.88095 mean HPG in sample including Vancouver
mean(y)    # Vancouver mean HPG: 6.88023
f=@formula(Y ~ 1 + tr)
m1 = reg(dd, f)
#tr          |  4.01087  
#(Intercept) |  2.86937 
sum(m1.coef) ≈ mean(y) # true
pr = FixedEffectModels.predict(m1, dd)[dd.tr.==1,:]
plot();
plot!(y, lab="Y(1)");
plot!(pr, lab="Y(0)")
#plot!(y, lab="Y(1)");

""
f=@formula(Y ~ 1 + tr + fe(tim))
m1 = reg(dd, f)
#tr          |  4.01087  # AZ same as before! 
pr = FixedEffectModels.predict(m1, dd)[dd.tr.==1,:]
plot();
plot!(y, lab="Y(1)");
plot!(pr, lab="Y(0)")
#plot!(y, lab="Y(1)");








f=@formula(Y ~ levels(loc)  + tr )
m1 = reg(dd, f)




f=@formula(Y ~ t + fe(loc) )
f=@formula(Y ~ t^2  )
f=@formula(Y ~ 1)
f=@formula(Y ~ fe(loc) + fe(tim) + pt1)

m1 = reg(dd, f)
m1.coef

fdid =@formula(Y ~ fe(loc) + fe(tim)
+ pt1 + pt2 + pt3 + pt4 + pt5 
+ pt6 + pt7 + pt8 + pt9 + pt10 
+ pt11 + pt12 + pt13 + pt14 + pt15 
+ pt16 + pt17 + pt18 + pt19 + pt20 
+ pt21 + pt22 + pt23 + pt24 + pt25 
#+ pt26 
       + pt27 + pt28 + pt29 + pt30 
+ pt31 + pt32 + pt33 + pt34 + pt35 
+ pt36 + pt37 + pt38 + pt39 + pt40 
+ pt41
)
#
#fieldnames(FixedEffectModel)
#m_did.coef
m_did = reg(dd, fdid, Vcov.cluster(:loc,:t))
m_did = reg(dd, fdid, Vcov.cluster(:loc))
m_did = reg(dd, fdid, Vcov.cluster(:t))
m_did = reg(dd, fdid, Vcov.robust())
m_did = reg(dd, fdid)


coef_did = [m_did.coef[1:25]; 0.0;  m_did.coef[26:40]]
ci = confint(m_did; level = 0.95)
ci_did = [ci[1:25,:]; zeros(1,2);  ci[26:40,:]]

# ATT := Y(1) - Y(0)
# Y(1) - ATT = Y(0) 
Y0hat = y .- coef_did
p1=plot(legend=:bottomleft);
plot!([T0] , seriestype = :vline, lab="", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red")  ;
plot!(coef_did, lab="ATT");
plot!(ci_did, l=:dash, c=2 ,lab="95% CI")
#
p2=plot(legend=:bottomleft)
plot!([T0] , seriestype = :vline, lab="Last quarter pre-treatment", color="red");
plot!([0.0],  seriestype = :hline, lab="", color="red");
plot!(y, c=1, lab="Y(1): Vancouver HPG Observed")
plot!(Y0hat, c=4, lab="Y(0): Synthetic Vancouver HPG")


plot(p2,p1, layout=(2,1),size=(600,400).*1.25)
plot(plt_Y0,plt_V, layout=(2,1),size=(600,400).*1.25)



plt_Y0 = plot(legend = :topleft)
plot!([T0], seriestype = :vline, lab="Last quarter pre-treatment", color="red")
plot!(time_ix, y, xticks=xt, c=1, lab="Y(1): Vancouver HPG observed")
plot!(time_ix, ŷ_L2,         c=4, lab="Y(0): EN Synthetic Vancouver HPG")
plot!(Y0hat, c=4, l=:dash, lab="Y(0): DiD Synthetic Vancouver HPG")
#plot!(time_ix, ŷ_L2, color=1, lab="Synthetic "*u_treated[i]) 
plot!([0.0],  seriestype = :hline, lab="", color="red")  # |> display




############################
fdid2 =@formula(Y ~ 1 + fe(loc) + fe(tim) 
#+ tr*t 
#+ t^2
#+ fe(loc)&t
+ pt1 + pt2 + pt3 + pt4 + pt5 
+ pt6 + pt7 + pt8 + pt9 + pt10 
+ pt11 + pt12 + pt13 + pt14 + pt15 
+ pt16 + pt17 + pt18 + pt19 + pt20 
+ pt21 + pt22 + pt23 + pt24 + pt25 
#+ pt26 
       + pt27 + pt28 + pt29 + pt30 
+ pt31 + pt32 + pt33 + pt34 + pt35 
+ pt36 + pt37 + pt38 + pt39 + pt40 
+ pt41
+ co*t
)
#
m_did = reg(dd, fdid2)
coef_did = [m_did.coef[1:25]; 0.0;  m_did.coef[26:40]]
ci = confint(m_did; level = 0.95)
ci_did = [ci[1:25,:]; zeros(1,2);  ci[26:40,:]]

# ATT := Y(1) - Y(0)
# Y(1) - ATT = Y(0) 
Y0hat = y .- coef_did
plot(legend=:bottomleft)
plot!([T0] , seriestype = :vline, lab="", color="red")
plot!([0.0],  seriestype = :hline, lab="", color="red")  
#
plot!(coef_did, lab="ATT")
plot!(ci_did, l=:dash, c=2 ,lab="95% CI")
#
plot!(y, lab="Y(1)")
#
plot!(Y0hat, lab="Y(0): Synthetic Vancouver HPG")






coef_did = [m_did.coef[1:25]; 0.0;  m_did.coef[26:40]]
Y0hat = y .- coef_did
plot(legend=:bottomleft)
plot!([T0] , seriestype = :vline, lab="", color="red")
plot!([0.0],  seriestype = :hline, lab="", color="red")  
plot!(y, lab="y")
plot!(coef_did, lab="ATT")
plot!(Y0hat, lab="Synthetic Y(0)")
