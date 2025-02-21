function [h1,parmHat1,kernelData] = plotWindDistribution(U,COLOR,varargin)

%% Inputparseer
p = inputParser();
p.CaseSensitive = false;
p.addOptional('plotOpt',1);
p.parse(varargin{:});
%%%%%%%%%%%%%%%%%%%%%%%%%%
plotOpt = p.Results.plotOpt ;

% U = newU_NORA3(indZ_NORA,:);
U(isnan(U)|isinf(U))=[];
maxU = nanmax(U);

h1 = [];

% vbins = 0:bindWidth:ceil(maxU);
% h1 = histogram(U(:), vbins,'Normalization','pdf');
% h1.FaceColor = COLOR;
% h1.FaceAlpha = 0.3;
% hold on

u = linspace(min(U),max(U),100);
pd1 = fitdist(U(:),'kernel');
Y1 = pdf(pd1,u);

if plotOpt ==1,
    h1(1) = plot(u,Y1,'color',COLOR,'linewidth',2);
    hold on
    xlabel('$\overline{u}$ (m s$^{-1}$)','interpreter','latex');
    ylabel('Probability density function');
    parmHat1 = wblfit(U(:));
    y = wblpdf(0:0.1:ceil(max(maxU)),parmHat1(1),parmHat1(2));
    h1(2) = plot(0:0.1:ceil(max(maxU)),y,'color','b','linewidth',2);
    set(gcf,'color','w')
else
    parmHat1 = wblfit(U(:));
    
end
legend('Kernel estimate','Weibull fit');

label(['a = ',num2str(round(parmHat1(1)*10)/10),', b = ',...
    num2str(round(parmHat1(2)*10)/10)],0.03,0.97,'alignement','left','verticalalignment','top');

kernelData.x = u;
kernelData.y = Y1;


end

