close all
fprintf(1,'\nStarting modify ARTdemo:\n\n');

% Inicializar matrices para almacenar errores de reconstrucción
errors_kaczmarz = [];
errors_symkaczmarz = [];
errors_randkaczmarz = [];
errors_sart = [];
errors_em = [];

ini = 1;
fin = 100;
step = 2;

%for i = [ini:step:fin]
  %fprintf(1,'i = %2.0d \n', i);

  % Set the parameters for the test problem.
  N = 50;           % The discretization points.
  p = 75;    % No. of parallel rays.
  eta = 0.05;       % Relative noise level.
  k = 50; % No. of iterations.
  nang = 36.0;   %nang = 1.0*i; % No. of used angles
  theta = 0:180.0/nang:179;

  %fprintf(1,'Creating a parallel-beam tomography test problem\n');
  %fprintf(1,'with N = %2.0f, theta = %1.0f:%1.0f:%3.0f, and p = %2.0f.',...
  %    [N,theta(1),theta(2)-theta(1),theta(end),p]);

  % Create the test problem.
  [A,b_ex,x_ex] = paralleltomo(N,theta,p);

  % Noise level.
  delta = eta*norm(b_ex);

  % Add noise to the rhs.
  randn('state',0);
  e = randn(size(b_ex));
  e = delta*e/norm(e);
  b = b_ex + e;

  % Show the exact solution.
  figure
  imagesc(reshape(x_ex,N,N)), colormap gray,
  axis image off
  c = caxis;
  title('Exact phantom')

  %fprintf(1,'\n\n');
  %fprintf(1,'Perform k = %2.0f iterations with FBP method.',k);
  %fprintf(1,'\nThis takes a moment ...');

  % Perform the kaczmarz iterations.
  tic
  Xfbp = fbp(A,b,theta,'ram-lak');
  toc

  %Show the fbp solution.
  figure
  imagesc(reshape(Xfbp,N,N)), colormap gray,
  axis image off
  caxis(c);
  title('FBP reconstruction')

  %fprintf(1,'\n\n');
  %fprintf(1,'Perform k = %2.0f iterations with Kaczmarz''s method.',k);
  %fprintf(1,'\nThis takes a moment ...');

  % Perform the kaczmarz iterations.
  %tic
  Xkacz = kaczmarz(A,b,k);
  %toc

  % Calcular error de reconstrucción para Kaczmarz
  error_kaczmarz = norm(x_ex - Xkacz, 'fro') / norm(x_ex, 'fro');
  errors_kaczmarz = [errors_kaczmarz error_kaczmarz];

  %Show the kaczmarz solution.
  %figure
  %imagesc(reshape(Xkacz,N,N)), colormap gray,
  %axis image off
  %caxis(c);
  %title('Kaczmarz reconstruction')

  %fprintf(1,'\n\n');
  %fprintf(1,'Perform k = %2.0f iterations with the symmetric Kaczmarz method.',k);
  %fprintf(1,'\nThis takes a moment ...');

  % Perform the symmetric kaczmarz iterations.
  %tic
  Xsymk = symkaczmarz(A,b,k);
  %toc

  % Calcular error de reconstrucción para Symmetric Kaczmarz
  error_symkaczmarz = norm(x_ex - Xsymk, 'fro') / norm(x_ex, 'fro');
  errors_symkaczmarz = [errors_symkaczmarz error_symkaczmarz];

  % Show the symmetric kaczmarz solution.
  %figure
  %imagesc(reshape(Xsymk,N,N)), colormap gray,
  %axis image off
  %caxis(c);
  %title('Symmetric Kaczmarz reconstruction')

  %fprintf(1,'\n\n');
  %fprintf(1,'Perform k = %2.0f iterations with the randomized Kaczmarz method.',k);
  %fprintf(1,'\nThis takes a moment ...\n');

  % Perform the randomized kaczmarz iterations.
  %tic
  Xrand = randkaczmarz(A,b,k);
  %toc

  % Calcular error de reconstrucción para Randomized Kaczmarz
  error_randkaczmarz = norm(x_ex - Xrand, 'fro') / norm(x_ex, 'fro');
  errors_randkaczmarz = [errors_randkaczmarz error_randkaczmarz];

  % Show the randomized kaczmarz solution.
  %figure
  %imagesc(reshape(Xrand,N,N)), colormap gray,
  %axis image off
  %caxis(c);
  %title('Randomized Kaczmarz reconstruction')

  %fprintf(1,'\n\n');
  %fprintf(1,'Perform k = %2.0f iterations with the SART method.',k);
  %fprintf(1,'\nThis takes a moment ...\n');

  % Perform the SART iterations.
  %tic
  Xsart = sart(A,b,k);
  %toc

  % Calcular error de reconstrucción para SART
  error_sart = norm(x_ex - Xsart, 'fro') / norm(x_ex, 'fro');
  errors_sart = [errors_sart error_sart];

  % Show the SART solution.
  %figure
  %imagesc(reshape(Xsart,N,N)), colormap gray,
  %axis image off
  %caxis(c);
  %title('SART reconstruction')

%endfor

% Graficar errores de reconstrucción para cada método en función de p
%figure
%plot([ini:step:fin]/10, errors_kaczmarz, '-o', 'DisplayName', 'Kaczmarz', 'Color', 'blue', 'MarkerFaceColor', 'blue');
%hold on
%plot([ini:step:fin]/10, errors_symkaczmarz, '-s', 'DisplayName', 'Kaczmarz simétrico', 'Color', 'red', 'MarkerFaceColor', 'red');
%plot([ini:step:fin]/10, errors_randkaczmarz, '-^', 'DisplayName', 'Kaczmarz aleatorio', 'Color', 'green', 'MarkerFaceColor', 'green');
%plot([ini:step:fin]/10, errors_sart, '-d', 'DisplayName', 'SART', 'Color', 'black', 'MarkerFaceColor', 'black');
%xlabel('Nivel de ruido \eta [%]')
%ylabel('Error de reconstrucción')
%title('Reconstruction Error vs. p for Different Methods')
%legend('Location', 'NorthEast')  % Colocar la leyenda en la esquina superior derecha
%grid on
%set(gca, 'FontSize', 20)


