### Plot a circle
print("Hello World")
c =  circle((0,0), 1, rgbcolor=(1,1,0), fill=True)
c.show()

### Plot cosinus
p1 = plot(cos, (-5,5))
p1.show()

### plot x^2
x = var('x')
show(plot(x^2, (x, -2, 2)))

### 2 plots
x = var('x')
regular = plot(x^2, (x,-2,2), color='purple')
skinny = plot(4*x^2, (x,-2,2), color='green')
show(regular + skinny)

### Parametric plot
x = var('x')
p2 = parametric_plot((cos(x),sin(x)^3), (x,0,2*pi), rgbcolor=hue(0.6))
p2.show()

### Several parametric plot
x = var('x')
p1 = parametric_plot((cos(x),sin(x)),(x,0,2*pi),rgbcolor=hue(0.2))
p2 = parametric_plot((cos(x),sin(x)^2),(x,0,2*pi),rgbcolor=hue(0.4))
p3 = parametric_plot((cos(x),sin(x)^3),(x,0,2*pi),rgbcolor=hue(0.6))
show(p1+p2+p3, axes=false)


### Make a polygon
L = [[-1+cos(pi*i/100)*(1+cos(pi*i/100)),
    2*sin(pi*i/100)*(1-cos(pi*i/100))] for i in range(200)]
p = polygon(L, rgbcolor=(1/8,3/4,1/2))
show(p, axes=false)

### Make another polygon
L = [[6*cos(pi*i/100)+5*cos((6/2)*pi*i/100),
     6*sin(pi*i/100)-5*sin((6/2)*pi*i/100)] for i in range(200)]
p = polygon(L, rgbcolor=(1/8,1/4,1/2))
t = text("hypotrochoid", (5,4), rgbcolor=(1,0,0))
show(p+t)

### arcsin
v = [(sin(x),x) for x in srange(-2*float(pi),2*float(pi),0.1)]
show(line(v))

### arctan
v = [(tan(x),x) for x in srange(-2*float(pi),2*float(pi),0.01)]
show(line(v), xmin=-20, xmax=20)

### contour plot
f = lambda x,y: cos(x*y)
c = contour_plot(f, (-4,4), (-4,4),cmap='hsv')
show(c)

### 3d plot
x, y = var('x,y')
p =plot3d(x^2 + y^2, (x,-2,2), (y,-2,2))
show(p)

### 3d paramettric plot
u, v = var('u, v')
f_x(u, v) = u
f_y(u, v) = v
f_z(u, v) = u^2 + v^2
p = parametric_plot3d([f_x, f_y, f_z], (u, -2, 2), (v, -2, 2))
show(p)

### yellow withney's umbrella
u, v = var('u,v')
fx = u*v
fy = u
fz = v^2
p = parametric_plot3d([fx, fy, fz], (u, -1, 1), (v, -1, 1),
                      frame=False, color="yellow")
show(p)

### cross cap
u, v = var('u,v')
fx = (1+cos(v))*cos(u)
fy = (1+cos(v))*sin(u)
fz = -tanh((2/3)*(u-pi))*sin(v)
p = parametric_plot3d([fx, fy, fz], (u, 0, 2*pi), (v, 0, 2*pi),
                  frame=False, color="red")
show(p)

### lemniscate
x, y, z = var('x,y,z')
f(x, y, z) = 4*x^2 * (x^2 + y^2 + z^2 + z) + y^2 * (y^2 + z^2 - 1)
p = implicit_plot3d(f, (x, -0.5, 0.5), (y, -1, 1), (z, -1, 1))
show(p)
