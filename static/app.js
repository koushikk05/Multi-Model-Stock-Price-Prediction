function logout(){
    fetch(`/logout`)
      .then(function (response) {
          if (response)
          window.location.href = response.url
      }).then(function (text) {
          console.log('GET response text:');
          console.log(text); 
      });
}