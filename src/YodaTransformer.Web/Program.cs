using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using YodaTransformer.Web;
using YodaTransformer.Web.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

builder.Services.AddScoped(sp =>
    new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
builder.Services.AddScoped<YodaModel>();

var host = builder.Build();

// Load the model before the first render so Chat works immediately.
var model = host.Services.GetRequiredService<YodaModel>();
await model.InitializeAsync();

await host.RunAsync();
