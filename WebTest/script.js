(function () {
      "use strict";

      var STORAGE_KEY = "deneme-blog-tema";
      var dugme = document.getElementById("tema-dugmesi");

      function temayiUygula(karanlik) {
        document.body.classList.toggle("dark", karanlik);
        if (dugme) {
          dugme.textContent = karanlik ? "Aydinlik Mod" : "Karanlik Mod";
          dugme.setAttribute("aria-pressed", karanlik ? "true" : "false");
        }
      }

      function kayitliTema() {
        try {
          return window.localStorage.getItem(STORAGE_KEY);
        } catch (e) {
          return null;
        }
      }

      function temayiKaydet(karanlik) {
        try {
          window.localStorage.setItem(STORAGE_KEY, karanlik ? "dark" : "light");
        } catch (e) {
          /* depolama kapali olabilir, sessizce gec */
        }
      }

      var baslangic = kayitliTema();
      if (baslangic === "dark") {
        temayiUygula(true);
      } else if (baslangic === "light") {
        temayiUygula(false);
      }

      if (dugme) {
        dugme.addEventListener("click", function () {
          var karanlik = !document.body.classList.contains("dark");
          temayiUygula(karanlik);
          temayiKaydet(karanlik);
        });
      }
    })();
    