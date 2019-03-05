importall POMDPs
using Distributions

type D4ICRA <: POMDPs.POMDP{Float64,Int,Int}
	discount_factor::Float64
	step_size::Int64
end



#2second = 50000 or maybe 1 second? 



function makeSolver()

	actions(::D4ICRA) = [0,1,2,3]
	n_actions(::D4ICRA) = 4
	pomdp = D4ICRA(0.9,1); 
	solver = POMCPSolver(tree_queries=50000, c=10)
	planner = solve(solver, pomdp);

	return [solver,pomdp,planner]; 

end


#observations, function of state only
function generate_o(p::D4ICRA, s::Array, a::Int, sp::Array, rng::AbstractRNG)

	#ROBOT_VIEW_RADIUS
	if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 20
		return 1
	else
		return 0
	end

end

#state transitions, function of state and action
function generate_sr(p::D4ICRA, s::Array, a::Int, rng::AbstractRNG)

	delta = 10; 
    if a == 0
        s = s + [0,delta,0,0]
    elseif a == 1
        s= s + [0,-delta,0,0]
    elseif a == 2
    	s= s + [delta,0,0,0]
    elseif a == 3
    	s= s + [-delta,0,0,0]
    else 
    	s= s
    end

    s[1] = max(0,s[1])
    s[1] = min(437,s[1]); 
    s[2] = max(0,s[2])
    s[2] = min(754,s[2]);

    r = 0; 


    if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 25
		r = 10
	else
		r = 0
	end

    return s,r
    
end





function initial_state_distribution(pomdp::D4ICRA)
    sig_0 = [10,10,sqrt(0.00001),sqrt(0.00001)]
    return MvNormal(sig_0)
end;


using BasicPOMCP
using POMDPToolbox
importall Base
using Dates



function getPOMCPAction(planner,b)
	return action(planner,b); 
end

actions(::D4ICRA) = [0,1,2,3]
n_actions(::D4ICRA) = 4
#solver,pomdp,planner = makeSolver(); 
#b = initial_state_distribution(pomdp); 
#act = getPOMCPAction(planner,b); 
#println(act); 
